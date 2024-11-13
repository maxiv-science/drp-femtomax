import logging
import os
import tempfile
import json
from dataclasses import dataclass

from dranspose.event import EventData
from dranspose.parameters import IntParameter, StrParameter, BoolParameter
from dranspose.middlewares.stream1 import parse
from dranspose.middlewares.sardana import parse as sardana_parse
from dranspose.data.stream1 import Stream1Data, Stream1Start, Stream1End
from dranspose.data.sardana import SardanaDataDescription
import numpy as np
from numpy import unravel_index
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import gaussian_filter, center_of_mass

logger = logging.getLogger(__name__)


def hithist(
    img, threshold_counting, pre_threshold
):  # This is the magic function that counts the photons using a 3x3 square (you can change the size of the box)
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


class IncompatibleImages(Exception):
    pass


class CleanImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0
    crop_left: int = 0

    def __add__(self, other: "CleanImage") -> "CleanImage":
        if self.image.shape == other.image.shape:
            if self.crop_top == other.crop_top and self.crop_right == other.crop_right and self.crop_bottom == other.crop_bottom and self.crop_left == other.crop_left:
                return CleanImage(image=self.image + other.image, crop_top=self.crop_top, crop_right=self.crop_right,
                                  crop_bottom=self.crop_bottom, crop_left=self.crop_left)
        raise IncompatibleImages()

    def to_dict(self):
        return {"image":self.image, "crop": {n[5:]: val for n, val in self.__dict__.items() if n.startswith("crop_")}  }

class WorkerResult(BaseModel):
    sardana: SardanaDataDescription | None = None
    means: dict[str, float] = {}
    image: CleanImage | None = None


class CmosWorker:
    @staticmethod
    def describe_parameters():
        params = [
            IntParameter(name="cmos_background", default=100, description="The initial background subtraction, similar to a dark image"),
            IntParameter(name="cmos_threshold", default=12, description="Only consider values above this threshold"),
            IntParameter(
                name="threshold_counting", default=108,
                description="important threshold for counting photons later"
            ),
            IntParameter(
                name="pre_threshold", default=1100,
                description="discard every spot with a total energy below that"
            ),
            StrParameter(name="rois", default="{}"),
        ]
        return params

    def __init__(self, **kwargs):
        self.number = 0
        self.accum = None

    def _cog(self, event, parameters, ret):
        logger.debug("cog using parameters %s", parameters)
        threshold_counting = parameters["threshold_counting"].value
        pre_threshold = parameters["pre_threshold"].value
        bg = parameters["background"].value

        dat = None
        if "andor3_balor" in event.streams:
            dat = parse(event.streams["andor3_balor"])
        elif "andor3_zyla10" in event.streams:
            dat = parse(event.streams["andor3_zyla10"])

        if isinstance(dat, Stream1Start):
            if isinstance(ret["sardana"], SardanaDataDescription):
                print("sardana start is", ret["sardana"])
                dstname = os.path.join(ret["sardana"].scandir, "process", "photoncount")
                for fn in ret["sardana"].scanfile:
                    if fn.endswith(".h5"):
                        name = f'{fn[:-3]}-{ret["sardana"].serialno}.h5'
                        dstname = os.path.join(dstname, name)
                        break
                return {**ret, "cog_filename": dstname}

        if isinstance(dat, Stream1End):
            return {**ret, "reconstructed": self.cumu}

        if not isinstance(dat, Stream1Data):
            return ret

        frame = dat.frame
        logger.debug("frameno %d", frame)
        hits = []
        try:
            rois = json.loads(parameters["rois"].value)
            if "cog" in rois:
                tl = rois["cog"]["handles"]["_handleBottomLeft"]
                br = rois["cog"]["handles"]["_handleTopRight"]

                xslice = slice(min(int(tl[0]), int(br[0])), max(int(tl[0]), int(br[0])))
                yslice = slice(min(int(tl[1]), int(br[1])), max(int(tl[1]), int(br[1])))

                crop = dat.data[yslice, xslice]

                hits = hithist(crop, threshold_counting, pre_threshold)
                if self.cumu is None:
                    self.cumu = np.zeros_like(dat.data)[yslice, xslice]

                for hit in hits:
                    self.cumu[int(hit[0]), int(hit[1])] += 1
                # print(hits)
        except Exception:
            pass
        dark_corr = dat.data.clip(min=bg) - bg
        return {**ret, "hits": hits, "frame": dat.frame, "img": dark_corr}

    def parse_rois(self, parameters):
        slice_rois = {}
        try:
            rois = json.loads(parameters["rois"].value)
            for roi_name in rois:
                tl = rois[roi_name]["handles"]["_handleBottomLeft"]
                br = rois[roi_name]["handles"]["_handleTopRight"]

                xslice = slice(min(int(tl[0]), int(br[0])), max(int(tl[0]), int(br[0])))
                yslice = slice(min(int(tl[1]), int(br[1])), max(int(tl[1]), int(br[1])))

                slice_rois[roi_name] = np.s_[yslice, xslice]
        except Exception:
            pass
        return slice_rois




    def process_event(self, event: EventData, parameters=None, tick=False, *args, **kwargs):
        ret = WorkerResult()
        if "sardana" in event.streams:
            ret.sardana = sardana_parse(event.streams["sardana"])

        rois = self.parse_rois(parameters)

        clean_image = None
        if "andor3_balor" in event.streams:
            data = parse(event.streams["andor3_balor"])

            if "photon" in rois:
                #TODO: if header, parse filename
                #TODO: if image, process roi and place into clean image
                if isinstance(data, Stream1Data):
                    clean_image = CleanImage(image=np.ones((100,100)))
                    #TODO: if too much photons fall back to normal cmos processing
            else:
                #TODO: self.process_cmos(data)
                pass
            clean_image = CleanImage(image= np.ones((100, 100)), )

        elif "andor3_zyla10" in event.streams:
            data = parse(event.streams["andor3_zyla10"])
            # TODO: self.process_cmos(data)
            clean_image = CleanImage(image= np.ones((100, 100)), )

        elif "pilatus" in event.streams:
            data = parse(event.streams["pilatus"])
            clean_image = CleanImage(image= np.ones((100, 100)), )

        if clean_image is not None:
            ret.image = clean_image

            means = {}
            for name, sli in rois.items():
                crop = clean_image.image[sli]
                scalar = crop.mean()
                means[name] = scalar
            ret.means = means

            if False:
                bg = parameters["background"].value
                if isinstance(dat, Stream1Start):
                    print("start message", dat)
                    return {**ret, "pileup_filename": dat.filename}
                if not isinstance(dat, Stream1Data):
                    return ret

                dark_corr = dat.data.clip(min=bg) - bg

                blur = parameters["blur"].value
                thr = parameters["threshold"].value
                sigma = parameters["sigma"].value
                if blur:
                    dark_corr[dark_corr < thr] = 0  # 12
                    dark_corr = gaussian_filter(dark_corr, sigma=sigma)

                means = {}
                try:
                    rois = json.loads(parameters["rois"].value)
                    for rn, rd in rois.items():
                        tl = rd["handles"]["_handleBottomLeft"]
                        br = rd["handles"]["_handleTopRight"]

                        xslice = slice(
                            min(int(tl[0]), int(br[0])), max(int(tl[0]), int(br[0]))
                        )
                        yslice = slice(
                            min(int(tl[1]), int(br[1])), max(int(tl[1]), int(br[1]))
                        )

                        crop = dark_corr[yslice, xslice]
                        scalar = crop.mean()
                        means[rn] = scalar
                except Exception:
                    pass

                return {"img": dark_corr, "cropped": None, "roi_means": means, **ret}

        return ret

    def finish(self, parameters=None):
        print("finished")
