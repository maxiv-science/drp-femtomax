import logging
import os
import tempfile
import json
from dataclasses import dataclass
from .peak_finding import findPeaks

from dranspose.event import EventData
from dranspose.parameters import (
    IntParameter,
    StrParameter,
    BoolParameter,
    FloatParameter,
)
from dranspose.middlewares.stream1 import parse
from dranspose.middlewares.sardana import parse as sardana_parse
from dranspose.middlewares.lecroy import parse as lecroy_parse
from dranspose.data.stream1 import Stream1Data, Stream1Start, Stream1End
from dranspose.data.sardana import SardanaDataDescription
from dranspose.data.lecroy import LecroyParsed
import numpy as np
from numpy import unravel_index
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import gaussian_filter, center_of_mass

from src.utils import parse_rois

logger = logging.getLogger(__name__)


class IncompatibleImages(Exception):
    pass


class CleanImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0
    crop_left: int = 0

    pixel_size: float | None = None  # in meters, we assume square pixels

    def set_crop_for_slices(self, orig: np.ndarray, slices: tuple[slice, slice]):
        self.crop_top = slices[0].start
        self.crop_bottom = orig.shape[0] - slices[0].stop
        self.crop_left = slices[1].start
        self.crop_right = orig.shape[1] - slices[1].stop

    def __add__(self, other: "CleanImage") -> "CleanImage":
        if self.image.shape == other.image.shape:
            if (
                self.crop_top == other.crop_top
                and self.crop_right == other.crop_right
                and self.crop_bottom == other.crop_bottom
                and self.crop_left == other.crop_left
            ):
                return CleanImage(
                    image=self.image + other.image,
                    crop_top=self.crop_top,
                    crop_right=self.crop_right,
                    crop_bottom=self.crop_bottom,
                    crop_left=self.crop_left,
                )
        raise IncompatibleImages(
            f"image of shape {self.image.shape} is incompatible with {other.image.shape}"
        )

    def to_dict(self):
        r = {
            "image": self.image,
            "crop": {
                n[5:]: val for n, val in self.__dict__.items() if n.startswith("crop_")
            },
        }
        if self.pixel_size is not None:
            r["pixel_size"] = [self.pixel_size, self.pixel_size]
        return r


class WorkerResult(BaseModel):
    sardana: SardanaDataDescription | None = None
    means: dict[str, float] = {}
    image: CleanImage | None = None
    photon_filename: str | None = None
    photon_xye: list[tuple[float, float, float]] = []
    photon_e_max: float | None = None
    frame_no: int | None = None
    osc_channels: list[int] | None = None
    osc_peak_pos: list[
        list[float]
    ] | None = None  # Channel[peaks] (don't care about trace number)
    osc_peak_amps: list[list[float]] | None = None  # Channel[peaks]
    osc_ntraces: list[int] | None = None  # Per worker - number of traces per channel
    osc_t_range: list[
        list[float]
    ] | None = None  # Per channel - time range endpoints of oscilloscope


class TooManyPhotons(Exception):
    pass


class CmosWorker:
    @staticmethod
    def describe_parameters():
        params = [
            IntParameter(
                name="cmos_background",
                default=100,
                description="The initial background subtraction, similar to a dark image",
            ),
            IntParameter(
                name="cmos_threshold",
                default=12,
                description="Only consider values above this threshold",
            ),
            IntParameter(
                name="threshold_counting",
                default=108,
                description="important threshold for counting photons later (this is with background)",
            ),
            IntParameter(
                name="pre_threshold",
                default=1100,
                description="discard every spot with a total energy below that",
            ),
            StrParameter(
                name="rois",
                default="{}",
            ),
            FloatParameter(
                name="threshold_low",
                default=0.006,
                description="Potential peaks of (-1)*osc data have a value above this (needed for parabolic fitting of peaks)",
            ),
            FloatParameter(
                name="threshold_high",
                default=0.035,
                description="Parabolic fit values for peaks of (-1)*osc data above this value are discarded",
            ),
        ]
        return params

    def __init__(self, **kwargs):
        self.number = 0
        self.accum = None

    def _cog(self, event, parameters, ret):
        if isinstance(event, Stream1Start):
            if isinstance(ret["sardana"], SardanaDataDescription):
                print("sardana start is", ret["sardana"])
                dstname = os.path.join(ret["sardana"].scandir, "process", "photoncount")
                for fn in ret["sardana"].scanfile:
                    if fn.endswith(".h5"):
                        name = f'{fn[:-3]}-{ret["sardana"].serialno}.h5'
                        dstname = os.path.join(dstname, name)
                        break
                return {**ret, "cog_filename": dstname}

    def photonize(self, img, threshold_counting, pre_threshold):
        """
        This is the magic function that counts the photons using a 3x3 square (you can change the size of the box)

        :param img: the image or cropped part
        :param threshold_counting: the value a pixel must surpass to be considered a hit
        :param pre_threshold: the intensity the blob in total has to have
        :return:
        """
        fullframe = img.clip(min=threshold_counting) - threshold_counting
        positions = np.asarray(fullframe > 0).nonzero()
        hits = set()
        for x, y in zip(*positions):
            small = fullframe[x - 1 : x + 2, y - 1 : y + 2]
            if np.prod(small.shape) == 0:
                continue

            maxpos = np.unravel_index(small.argmax(), small.shape)
            hits.add((x + maxpos[0] - 1, y + maxpos[1] - 1))
        pos = []
        for x, y in hits:
            hit = fullframe[x - 1 : x + 2, y - 1 : y + 2]
            d1 = hit.sum()
            cm = center_of_mass(hit)
            pos.append((x + cm[0] - 1, y + cm[1] - 1, d1))
        return [p for p in pos if p[2] > pre_threshold]

    def process_cmos(self, img, cmos_background, cmos_threshold):
        dark_corr = img.clip(min=cmos_background) - cmos_background
        dark_corr[dark_corr < cmos_threshold] = 0
        return dark_corr

    def process_event(
        self, event: EventData, parameters=None, tick=False, *args, **kwargs
    ):
        ret = WorkerResult()
        if "sardana" in event.streams:
            ret.sardana = sardana_parse(event.streams["sardana"])

        rois = parse_rois(parameters)

        clean_image = None
        if "andor3_balor" in event.streams:
            data = parse(event.streams["andor3_balor"])

            if "photon" in rois:
                if isinstance(data, Stream1Start):
                    if data.filename != "":
                        ret.photon_filename = data.filename
                if isinstance(data, Stream1Data):
                    ret.frame_no = data.frame
                    roi = rois["photon"]
                    crop = data.data[roi]
                    threshold_counting = parameters["threshold_counting"].value
                    pre_threshold = parameters["pre_threshold"].value
                    hits = self.photonize(crop, threshold_counting, pre_threshold)
                    clean = np.zeros_like(data.data)[roi]

                    for hit in hits:
                        clean[int(hit[0]), int(hit[1])] += 1
                    clean_image = CleanImage(image=clean, pixel_size=12e-6)
                    clean_image.set_crop_for_slices(data.data, roi)
                    logger.info("in worker is %s", clean_image)
                    ret.photon_xye = [
                        (x + clean_image.crop_top, y + clean_image.crop_left, e)
                        for x, y, e in hits
                    ]
                    ret.photon_e_max = max([0] + [e for _, _, e in hits])

            else:
                if isinstance(data, Stream1Data):
                    cmos_background = parameters["cmos_background"].value
                    cmos_threshold = parameters["cmos_threshold"].value
                    clean_image = CleanImage(
                        image=self.process_cmos(
                            data.data, cmos_background, cmos_threshold
                        ),
                        pixel_size=12e-6,
                    )
                    ret.frame_no = data.frame

        elif "andor3_zyla10" in event.streams:
            data = parse(event.streams["andor3_zyla10"])
            if isinstance(data, Stream1Data):
                cmos_background = parameters["cmos_background"].value
                cmos_threshold = parameters["cmos_threshold"].value
                clean_image = CleanImage(
                    image=self.process_cmos(data.data, cmos_background, cmos_threshold),
                    pixel_size=6.5e-6,
                )
                ret.frame_no = data.frame

        elif "pilatus" in event.streams:
            data = parse(event.streams["pilatus"])
            if isinstance(data, Stream1Data):
                clean_image = CleanImage(image=data.data, pixel_size=172e-6)
                ret.frame_no = data.frame

        elif "andor3_zyla12" in event.streams:
            data = parse(event.streams["andor3_zyla12"])
            if isinstance(data, Stream1Data):
                cmos_background = parameters["cmos_background"].value
                cmos_threshold = parameters["cmos_threshold"].value
                clean_image = CleanImage(
                    image=self.process_cmos(data.data, cmos_background, cmos_threshold),
                    pixel_size=6.5e-6,
                )
                ret.frame_no = data.frame

        if "oscc_02_maui" in event.streams:
            osc = lecroy_parse(event.streams["oscc_02_maui"])
            if isinstance(osc, LecroyParsed):
                ret.osc_channels = osc.channels
                ret.osc_peak_pos = []
                ret.osc_peak_amps = []
                ret.osc_ntraces = []
                ret.osc_t_range = []

                for ch_traces, ch_meta in zip(osc.data, osc.meta):
                    allpos = []
                    allamps = []
                    time_base = (
                        np.arange(ch_traces.shape[-1]) * ch_meta.horiz_interval
                        + ch_meta.horiz_offset
                    )
                    ret.osc_t_range.append([time_base[0], time_base[-1]])
                    for trace in ch_traces:
                        peaks, amps, _, _ = findPeaks(
                            parameters["threshold_low"].value,
                            parameters["threshold_high"].value,
                            (-1.0) * trace,
                            time_base,
                        )
                        allpos += peaks
                        allamps += amps
                    ret.osc_peak_pos.append(
                        allpos
                    )  # Peak positions of all traces from a single worker, per channel
                    ret.osc_peak_amps.append(allamps)
                    ret.osc_ntraces.append(len(ch_traces))

        if clean_image is not None:
            ret.image = clean_image

            means = {}
            for name, sli in rois.items():
                crop = clean_image.image[sli]
                scalar = crop.clip(min=0).mean()
                means[name] = scalar
            ret.means = means

        return ret

    def finish(self, parameters=None):
        print("finished")
