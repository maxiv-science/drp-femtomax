import json
import logging
from pathlib import Path
from typing import Any

from dranspose.event import ResultData
from dranspose.parameters import StrParameter, BoolParameter
import os
import h5py
import numpy as np
from datetime import datetime

from pydantic import BaseModel

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
        ]
        return params

    def __init__(self, parameters=None, context=None, state=None, **kwargs):
        self._fh = None
        self.publish = {}

        self.pileup_filename = None

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

    def _setup_cog_file(self, result, parameters):
        self.cog_filename = result.payload["cog_filename"]
        if self._fh is None:
            # parts = self.cog_filename.split(".")
            fn = self.cog_filename
            logger.info("write to %s", fn)
            # fn = "./process/photoncount/testoutput.h5"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            self._fh = h5py.File(fn, "w")
            group = self._fh.create_group("hits")
            threshold_counting = parameters["threshold_counting"].value
            pre_threshold = parameters["pre_threshold"].value
            try:
                rois = json.loads(parameters["rois"].value)
                if "cog" in rois:
                    tl = rois["cog"]["handles"]["_handleBottomLeft"]
                    br = rois["cog"]["handles"]["_handleTopRight"]

                    xslice = slice(
                        min(int(tl[0]), int(br[0])), max(int(tl[0]), int(br[0]))
                    )
                    yslice = slice(
                        min(int(tl[1]), int(br[1])), max(int(tl[1]), int(br[1]))
                    )
                    group.create_dataset("roi_x", data=[xslice.start, xslice.stop])
                    group.create_dataset("roi_y", data=[yslice.start, yslice.stop])
            except Exception:
                pass

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
            self._fh["raw_data"] = h5py.ExternalLink(self.cog_filename, "/")

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

            self.publish.update(self.accum.to_dict())

        if len(res.photon_xye) > 0:
            logger.info("photons %s", res.photon_xye)

        if res.photon_e_max is not None:
            self.publish["max_e"].append(res.photon_e_max)

        return
        if "sardana" in result.payload:
            if result.payload["sardana"] is not None:
                self.publish["sardana"][result.event_number] = result.payload[
                    "sardana"
                ].model_dump()

        if "pileup_filename" in result.payload:
            self.pileup_filename = result.payload["pileup_filename"]

        if "cog_filename" in result.payload:
            self._setup_cog_file(result, parameters)

        # if analysis_mode == "roi":
        if "img" in result.payload:
            img = result.payload["img"]

            if parameters["integrate"].value:
                if self.publish["last"].shape != img.shape:
                    self.publish["last"] = img
                    self.context["last"] = img
                    self.publish["nint"] = 1
                else:
                    self.publish["last"] = self.publish["last"] + img
                    self.context["last"] = self.publish["last"]
                    self.publish["nint"] += 1
            else:
                self.publish["last"] = img
                self.context["last"] = img
                self.publish["nint"] = 1
            if "roi_means" in result.payload:
                mean = result.payload["roi_means"]
                self.publish["roi_means"][result.event_number] = mean

            if self.allsum is None:
                self.allsum = img
                self.nimg = 1
            else:
                self.allsum = self.allsum + img
                self.nimg += 1

        if analysis_mode == "sparsification":
            if result.payload:
                self.publish["hits"][result.event_number] = result.payload
                if self.dset:
                    nhits = result.payload["offsets"].shape[0]
                    oldsize = self.dset.shape[0]
                    self.dset.resize(oldsize + nhits, axis=0)
                    self.offsetdset.resize(oldsize + nhits, axis=0)

                    self.dset[oldsize : oldsize + nhits, :, :] = result.payload["spots"]
                    self.offsetdset[oldsize : oldsize + nhits, :] = result.payload[
                        "offsets"
                    ]
                    logger.debug("written record")

        elif analysis_mode == "cog":
            if "hits" in result.payload:
                hits_fr = [result.payload["frame"]] * len(result.payload["hits"])

                if self.xye_dset is not None:
                    oldsize = self.xye_dset.shape[0]
                    self.xye_dset.resize(oldsize + len(hits_fr), axis=0)
                    self.fr_dset.resize(oldsize + len(hits_fr), axis=0)
                    self.xye_dset[oldsize : oldsize + len(hits_fr), :] = result.payload[
                        "hits"
                    ]
                    self.fr_dset[oldsize : oldsize + len(hits_fr)] = hits_fr
            if "reconstructed" in result.payload:
                if self._fh is not None:
                    self._fh["hits"].create_dataset(
                        "image", data=result.payload["reconstructed"]
                    )

    def finish(self, parameters=None):
        logger.info("finished reducer custom, %s", self.publish)
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
        if self._fh:
            self._fh.close()
