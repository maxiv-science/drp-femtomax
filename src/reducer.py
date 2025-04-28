import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from dranspose.event import ResultData
from dranspose.data.sardana import SardanaDataDescription, SardanaRecordData
from dranspose.parameters import StrParameter, BoolParameter, IntParameter
import os
import h5py
import numpy as np
from datetime import datetime

from pydantic import BaseModel

from src.utils import parse_rois
from src.worker import WorkerResult, IncompatibleImages, CleanImage

logger = logging.getLogger(__name__)


class Accumulator(BaseModel):
    image: CleanImage | None = None
    number: int = 1

    def add_image(self, image):
        self.image = self.image + image
        self.number += 1

    def to_dict(self):
        return {**self.image.to_dict(), "accumulated_number": self.number}


class CmosReducer:
    @staticmethod
    def describe_parameters():
        params = [
            BoolParameter(name="integrate"),
            BoolParameter(name="pileup"),
            IntParameter(
                name="nbins",
                default=100,
                description="Number of bins for binning the osc peak positions in bins, determined by the number of bins, nbins",
            ),
            IntParameter(
                name="samplesize",
                default=100,
                description="The average number of osc peaks per trace is calculated using the last N=samplesize traces",
            ),
        ]
        return params

    def __init__(self, parameters=None, context=None, state=None, **kwargs):
        self.fh_lock = Lock()
        self._fh = None

        self.pileup_filename = None
        self.pile = None
        self.pile_fail = False

        self.state = state

        self.xye_dset = None

        self.context = context

        self.ref_movable = None

        self.movable_positions = {}

        if "prev_accumulator" in context:
            self.accum = context["prev_accumulator"]
        else:
            self.accum = Accumulator(image=CleanImage(image=np.zeros((100, 100))))

        self.count_sample = {}

        self.publish: dict[str, Any] = {"roi_means": {}, "step_means": {}, "max_e": []}
        self.publish.update(self.accum.to_dict())

        self.publish["sardana"] = {}

        self.publish["osc"] = {}

    def _setup_photon_file(self, filename, parameters):
        with self.fh_lock:
            if self._fh is None:
                logger.info("write to %s", filename)
                # fn = "./process/photoncount/testoutput.h5"
                path, fname = os.path.split(filename)
                root = os.path.splitext(fname)[0]
                fname = "%s_integrated.h5" % root
                output_folder = path.replace("raw", "process/xye")
                if output_folder != "":
                    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
                output_filename = os.path.join(output_folder, fname)

                self._fh = h5py.File(output_filename, "w")
                group = self._fh.create_group("hits")
                threshold_counting = parameters["threshold_counting"].value
                pre_threshold = parameters["pre_threshold"].value

                rois = parse_rois(parameters)
                if "photon" in rois:
                    roi = rois["photon"]
                    group.create_dataset("roi_x", data=[roi[1].start, roi[1].stop])
                    group.create_dataset("roi_y", data=[roi[0].start, roi[0].stop])

                group.create_dataset("pre_threshold", data=pre_threshold)
                group.create_dataset("threshold_counting", data=threshold_counting)
                meta = group.create_group("meta")
                try:
                    meta.create_dataset(
                        "dranspose_version", data=str(self.state.dranspose_version)
                    )
                    meta.create_dataset(
                        "mapreduce_commit_hash",
                        data=str(self.state.mapreduce_version.commit_hash),
                    )
                    meta.create_dataset(
                        "mapreduce_branch_name",
                        data=str(self.state.mapreduce_version.branch_name),
                    )
                    meta.create_dataset(
                        "mapreduce_timestamp",
                        data=str(self.state.mapreduce_version.timestamp),
                    )
                    meta.create_dataset(
                        "mapreduce_repository_url",
                        data=str(self.state.mapreduce_version.repository_url),
                    )
                except Exception:
                    pass

                self.xye_dset = group.create_dataset(
                    "hits_xye", (0, 3), maxshape=(None, 3), dtype=np.float64
                )
                self.fr_dset = group.create_dataset(
                    "hits_frame_number", (0,), maxshape=(None,), dtype=np.uint32
                )
                self._fh["raw_data"] = h5py.ExternalLink(filename, "/")

    def write_xye(self, frame_no, photon_xye):
        hits_fr = [frame_no] * len(photon_xye)
        with self.fh_lock:
            if self.xye_dset is not None:
                oldsize = self.xye_dset.shape[0]
                self.xye_dset.resize(oldsize + len(hits_fr), axis=0)
                self.fr_dset.resize(oldsize + len(hits_fr), axis=0)
                self.xye_dset[oldsize : oldsize + len(hits_fr), :] = photon_xye
                self.fr_dset[oldsize : oldsize + len(hits_fr)] = hits_fr

    def _update_rois(self, res):
        for roi in res.means:
            if roi not in self.publish["roi_means"]:
                self.publish["roi_means"][roi] = []
            cur_len = len(self.publish["roi_means"][roi])
            if cur_len <= res.frame_no:
                self.publish["roi_means"][roi] += [0] * (res.frame_no - cur_len + 1)
            self.publish["roi_means"][roi][res.frame_no] = res.means[roi]

            if roi not in self.publish["step_means"]:
                self.publish["step_means"][roi] = {
                    "y": [],
                    "y_errors": [],
                    "x": [],
                    "x_attrs": {"long_name": self.ref_movable},
                }
                self.publish["step_means"][roi + "_attrs"] = {
                    "NX_class": "NXdata",
                    "signal": "y",
                    "axes": ["x"],
                }
            last = 0
            for pnum, pos in enumerate(self.movable_positions):
                if pnum < len(self.publish["step_means"][roi]["y"]):
                    last = pos + 1
                    continue
                use = self.publish["roi_means"][roi][last : pos + 1]
                logger.info("use to mean for pos %s: %s", pos, use)
                mean = np.mean(use)
                std = np.std(use)
                self.publish["step_means"][roi]["y"].append(mean)
                self.publish["step_means"][roi]["y_errors"].append(std)
                self.publish["step_means"][roi]["x"].append(self.movable_positions[pos])
                last = pos + 1

    def _update_histogram(self, res, parameters=None):
        if res.osc_peak_pos is not None and res.osc_peak_amps is not None:
            samplesize = parameters["samplesize"].value
            for osc_ch, osc_pos, osc_amps, osc_ntraces in zip(
                res.osc_channels, res.osc_peak_pos, res.osc_peak_amps, res.osc_ntraces
            ):
                ch_str = "channel_0" + str(osc_ch)
                if ch_str not in self.publish["osc"]:
                    self.publish["osc"][ch_str] = {}
                    self.publish["osc"][ch_str]["pos"] = []
                    self.publish["osc"][ch_str]["amps"] = []
                    self.publish["osc"][ch_str]["hist"] = {
                        "x": None,
                        "y": np.zeros(parameters["nbins"].value),
                    }
                    self.publish["osc"][ch_str]["hist_attrs"] = {
                        "NX_class": "NXdata",
                        "axes": ["x"],
                        "signal": "y",
                    }

                    self.publish["osc"][ch_str]["ave_npeak_per_trace"] = []
                    self.publish["osc"][ch_str]["true_samplesize"] = []
                    self.count_sample[
                        ch_str
                    ] = (
                        []
                    )  # BUG: the "last" N traces are not chronologically last, but the last N given by some random workers that happened to be faster
                self.publish["osc"][ch_str]["pos"] += osc_pos
                self.publish["osc"][ch_str]["amps"] += osc_amps
                counts, bin_edges = np.histogram(
                    osc_pos, bins=parameters["nbins"].value
                )
                self.publish["osc"][ch_str]["hist"]["x"] = (
                    bin_edges[:-1] + np.diff(bin_edges) / 2.0
                )  # Bin centers
                self.publish["osc"][ch_str]["hist"]["y"] += counts

                count_ave_singleworker = sum(counts) / osc_ntraces
                self.count_sample[ch_str] += [count_ave_singleworker] * osc_ntraces

                self.count_sample[ch_str] = self.count_sample[ch_str][-samplesize:]
                count_ave = np.mean(self.count_sample[ch_str])
                true_size = min(len(self.count_sample[ch_str]), samplesize)
                self.publish["osc"][ch_str]["ave_npeak_per_trace"].append(count_ave)
                self.publish["osc"][ch_str]["true_samplesize"].append(true_size)

    def process_result(self, result: ResultData, parameters=None):
        res: WorkerResult = result.payload

        if res.image is not None:
            if parameters["integrate"].value:
                try:
                    self.accum.add_image(res.image)
                except IncompatibleImages:
                    self.accum = Accumulator(image=res.image)
            else:
                self.accum = Accumulator(image=res.image)

            self.context["prev_accumulator"] = self.accum
            self.publish.update(self.accum.to_dict())

            if parameters["pileup"].value and not self.pile_fail:
                if self.pile is None:
                    self.pile = Accumulator(image=res.image)
                else:
                    try:
                        self.pile.add_image(res.image)
                    except IncompatibleImages:
                        self.pile_fail = True

        if res.photon_filename is not None:
            # this is a start event, and we need to create the file
            # this result might be processed from multiple workers, so we need a lock
            self._setup_photon_file(res.photon_filename, parameters)

        if len(res.photon_xye) > 0:
            logger.debug("photons %s", res.photon_xye)
            self.write_xye(res.frame_no, res.photon_xye)

        if res.photon_e_max is not None:
            self.publish["max_e"].append(res.photon_e_max)

        if res.sardana:
            if isinstance(res.sardana, SardanaDataDescription):
                logger.info("start of sardana movable: %s", res.sardana.ref_moveables)
                if len(res.sardana.ref_moveables) > 0:
                    self.ref_movable = res.sardana.ref_moveables[0]
            if isinstance(res.sardana, SardanaRecordData):
                if self.ref_movable is not None:
                    logger.info(
                        "got position %s", getattr(res.sardana, self.ref_movable)
                    )
                    self.movable_positions[res.frame_no] = getattr(
                        res.sardana, self.ref_movable
                    )

        self._update_rois(res)

        self._update_histogram(res, parameters)

    def finish(self, parameters=None):
        logger.info("finished reducer custom")
        if self.pile:
            pass

        if False:
            try:
                pileup = parameters["pileup"].value
            except Exception as e:
                print(e.__repr__())
                pileup = False
            print("sumup to ", self.pileup_filename, pileup, parameters["pileup"].data)
            if pileup and self.pileup_filename:
                if self.pileup_filename.endswith(".h5"):
                    filename = self.pileup_filename[:-3] + "_pileup.h5"
                else:
                    filename = self.pileup_filename + "_pileup.h5"

                if os.path.isfile(filename):
                    logger.error("file exists already, adding time suffix")
                    filename += datetime.now().isoformat() + ".h5"
                print("opening file", Path(filename))
                self._fh = h5py.File(filename, "w")
                group = self._fh.create_group("pileup")
                print("group is", group)
                self.nimagesdset = group.create_dataset("nimages", data=self.nimg)
                self.dset = group.create_dataset("data", data=self.allsum)
        with self.fh_lock:
            if self._fh is not None:
                self._fh.close()
