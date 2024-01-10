import logging

from dranspose.event import ResultData
from dranspose.parameters import StrParameter
import os
import h5py
import numpy as np

logger = logging.getLogger(__name__)

class CmosReducer:
    @staticmethod
    def describe_parameters():
        params = [
            StrParameter(name="filename"),
        ]
        return params

    def __init__(self, parameters=None):
        self.publish = {"hits": {}}
        try:
            filename = parameters["filename"].data
        except:
            filename = None
        size = int(parameters["spot_size"].data)
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
        else:
            self._fh = None

    def process_result(self, result: ResultData, parameters=None):
        if result.payload:
            self.publish["hits"][result.event_number] = result.payload
            if self.dset:
                logger.debug("write dataset to file")
                nhits = result.payload["offsets"].shape[0]
                oldsize = self.dset.shape[0]
                self.dset.resize(oldsize + nhits, axis=0)
                self.offsetdset.resize(oldsize + nhits, axis=0)

                self.dset[oldsize:oldsize+nhits,:,:] = result.payload["spots"]
                self.offsetdset[oldsize:oldsize+nhits,:] = result.payload["offsets"]

    def finish(self, parameters=None):
        logger.info("finished reducer")
        if self._fh:
            self._fh.close()
