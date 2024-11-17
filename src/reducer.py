import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from dranspose.event import ResultData
from dranspose.parameters import StrParameter, BoolParameter
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

        if "prev_accumulator" in context:
            self.accum = context["prev_accumulator"]
        else:
            self.accum = Accumulator(image=CleanImage(image=np.zeros((100, 100))))

        self.publish: dict[str, Any] = {"roi_means": {}, "max_e":[]}
        self.publish.update(self.accum.to_dict())

        self.publish["sardana"] = {}

    def _setup_photon_file(self, filename, parameters):
        with self.fh_lock:
            if self._fh is None:
                logger.info("write to %s", filename)
                # fn = "./process/photoncount/testoutput.h5"
                path, fname = os.path.split(filename)
                root = os.path.splitext(fname)[0]
                fname = '%s_integrated.h5' % root
                output_folder = path.replace('raw', 'process/xye')
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
            hits_fr = [res.frame_no] * len(res.photon_xye)

            with self.fh_lock:
                if self.xye_dset is not None:
                    oldsize = self.xye_dset.shape[0]
                    self.xye_dset.resize(oldsize + len(hits_fr), axis=0)
                    self.fr_dset.resize(oldsize + len(hits_fr), axis=0)
                    self.xye_dset[oldsize: oldsize + len(hits_fr), :] = res.photon_xye
                    self.fr_dset[oldsize: oldsize + len(hits_fr)] = hits_fr


        if res.photon_e_max is not None:
            self.publish["max_e"].append(res.photon_e_max)

        if res.sardana:
            logger.info("sardana %s", res.sardana)

        for roi in res.means:
            if roi not in self.publish["roi_means"]:
                self.publish["roi_means"][roi] = []
            cur_len = len(self.publish["roi_means"][roi])
            if cur_len <= res.frame_no:
                self.publish["roi_means"][roi] += [0]*(res.frame_no-cur_len+1)
            self.publish["roi_means"][roi][res.frame_no] = res.means[roi]

        return
        if "sardana" in result.payload:
            if result.payload["sardana"] is not None:
                self.publish["sardana"][result.event_number] = result.payload[
                    "sardana"
                ].model_dump()


        elif analysis_mode == "cog":
            if "reconstructed" in result.payload:
                if self._fh is not None:
                    self._fh["hits"].create_dataset(
                        "image", data=result.payload["reconstructed"]
                    )

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
