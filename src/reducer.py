import logging
from pathlib import Path

from dranspose.event import ResultData
from dranspose.parameters import StrParameter, BoolParameter
import os
import h5py
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class CmosReducer:
    @staticmethod
    def describe_parameters():
        params = [
            StrParameter(name="filename"),
            BoolParameter(name="pileup"),
            StrParameter(name="scandir"),
            BoolParameter(name="integrate"),
        ]
        return params

    def __init__(self, parameters=None,  context=None,**kwargs):
        self._fh = None
        self.publish = {}
        self.allsum = None
        self.nimg = None
        self.pileup_filename = None

        if "analysis_mode" in parameters:
            analysis_mode = parameters["analysis_mode"].value
        else:
            analysis_mode = "roi"

        self.context = context
        if analysis_mode == "roi":
            if "last" in context:
                last = context["last"]
            else:
                last = np.zeros((100,100))
            self.publish = {"last": last, "cropped": None, "nint": 0, "roi_means":{}}


        elif analysis_mode == "sparsification":
            self.publish = {"hits": {}}
            try:
                filename = parameters["filename"].value
            except:
                filename = None
            size = parameters["spot_size"].value
            logger.info("writing to file %s", filename)
            self.dset = None
            if filename:
                if os.path.isfile(filename):
                    self._fh = h5py.File(filename, 'a')
                    self.offsetdset = self._fh.get("sparse/offset")
                    self.dset = self._fh.get("sparse/data")
                else:
                    self._fh = h5py.File(filename, 'w')
                    group = self._fh.create_group("sparse")
                    self.offsetdset = group.create_dataset("offset", (0,3), maxshape=(None,3), dtype=np.int16)
                    self.dset = group.create_dataset("data", (0,size, size), maxshape=(None,size, size), dtype=np.uint16)
                    self._offset_dset_name = f"sparse/offset"
                logger.info("opened file at %s", self._fh)

        self.publish["sardana"] = {}

    def process_result(self, result: ResultData, parameters=None):
        if result.payload is None:
            return

        if "analysis_mode" in parameters:
            analysis_mode = parameters["analysis_mode"].value
        else:
            analysis_mode = "roi"

        if "sardana" in result.payload:
            if result.payload["sardana"] is not None:
                self.publish["sardana"][result.event_number] = result.payload["sardana"].model_dump()

        if "pileup_filename" in result.payload:
            self.pileup_filename = result.payload["pileup_filename"]

        if analysis_mode == "roi":
            if "img" in result.payload:
                img = result.payload["img"]
                cropped = result.payload["cropped"]
                mean = result.payload["roi_means"]
                if parameters["integrate"].value:
                    if self.publish["last"].shape != img.shape or cropped != self.publish["cropped"]:
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
                self.publish["cropped"] = cropped
                self.publish["roi_means"][result.event_number] = mean

                if self.allsum is None:
                    self.allsum = img
                    self.nimg = 1
                else:
                    self.allsum = self.allsum + img
                    self.nimg += 1

        elif analysis_mode == "sparsification":
            if result.payload:
                self.publish["hits"][result.event_number] = result.payload
                if self.dset:
                    nhits = result.payload["offsets"].shape[0]
                    oldsize = self.dset.shape[0]
                    self.dset.resize(oldsize + nhits, axis=0)
                    self.offsetdset.resize(oldsize + nhits, axis=0)

                    self.dset[oldsize:oldsize+nhits,:,:] = result.payload["spots"]
                    self.offsetdset[oldsize:oldsize+nhits,:] = result.payload["offsets"]
                    logger.debug("written record")

    def finish(self, parameters=None):
        print(self.publish)
        logger.info("finished reducer custom")
        print(self.allsum)
        if self.allsum is not None:
            try:
                pileup = parameters["pileup"].value
            except Exception as e:
                print(e.__repr__())
                pileup=False
            print("sumup to ", self.pileup_filename, pileup, parameters["pileup"].data)
            if pileup and self.pileup_filename:
                if self.pileup_filename.endswith(".h5"):
                    filename = self.pileup_filename[:-3] + "_pileup.h5"
                else:
                    filename = self.pileup_filename + "_pileup.h5"

                if os.path.isfile(filename):
                    logger.error("file exists already, adding time suffix")
                    filename+=datetime.now().isoformat()+".h5"
                print("opening file", Path(filename))
                self._fh = h5py.File(filename, 'w')
                group = self._fh.create_group("pileup")
                print("group is", group)
                self.nimagesdset = group.create_dataset("nimages", data=self.nimg)
                self.dset = group.create_dataset("data", data=self.allsum)
        if self._fh:
            self._fh.close()
