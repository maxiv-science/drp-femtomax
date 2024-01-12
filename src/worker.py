import logging
import tempfile

from dranspose.event import EventData
from dranspose.parameters import IntParameter, StrParameter
from dranspose.middlewares.stream1 import parse
from dranspose.data.stream1 import Stream1Data
import numpy as np
from numpy import unravel_index

logger = logging.getLogger(__name__)


class CmosWorker:

    @staticmethod
    def describe_parameters():
        params = [
            IntParameter(name="background", default=0),
            IntParameter(name="threshold"),
            IntParameter(name="spot_size"),
            StrParameter(name="analysis_mode", default="roi"),
        ]
        return params

    def __init__(self, **kwargs):
        self.number = 0

    def process_event(self, event: EventData, parameters=None):

        if parameters["analysis_mode"].value == "roi":
            if "andor3_balor" in event.streams:
                dat = parse(event.streams["andor3_balor"])
                bg = parameters["background"].value
                if not isinstance(dat, Stream1Data):
                    return
                return {"img": dat.data.clip(min=bg) - bg , "cropped": None}

        elif parameters["analysis_mode"].data == b"sparsification":
            logger.debug("using parameters %s", parameters)
            bg = int(parameters["background"].data)
            thr = int(parameters["threshold"].data)
            size = int(parameters["spot_size"].data)
            dat = parse(event.streams["andor3_balor"])
            if not isinstance(dat, Stream1Data):
                return
            dark_corr = dat.data.clip(min=bg) - bg
            pad = int(np.ceil(size / 2))
            padded = np.pad(dark_corr, pad)

            frame = dat.frame
            logger.debug("frameno %d", frame)
            positions = np.asarray(padded > thr).nonzero()
            hits = set()
            for x, y in zip(*positions):
                small = padded[x - 1:x + 2, y - 1:y + 2]
                maxpos = unravel_index(small.argmax(), small.shape)
                hits.add((x + maxpos[0] - 1, y + maxpos[1] - 1))

            logger.debug("found %d hits", len(hits))
            spots = np.zeros((len(hits), size,size))
            offsets = np.zeros((len(hits), 3), dtype=np.int16)
            for i, (x,y) in enumerate(hits):
                spot = padded[pad + x - 3:pad + x + 4, pad + y - 3:pad + y + 4]
                spots[i] = spot
                offsets[i] = [frame, x - 3, y - 3]
            return {"spots": spots, "offsets": offsets}

    def finish(self, parameters=None):
        print("finished")
