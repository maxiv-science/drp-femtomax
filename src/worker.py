import logging
import tempfile
import json

from dranspose.event import EventData
from dranspose.parameters import IntParameter, StrParameter
from dranspose.middlewares.stream1 import parse
from dranspose.middlewares.sardana import parse as sardana_parse
from dranspose.data.stream1 import Stream1Data, Stream1Start
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
            StrParameter(name="rois", default=""),
        ]
        return params

    def __init__(self, **kwargs):
        self.number = 0

    def process_event(self, event: EventData, parameters=None):
        ret = {}
        if "sardana" in event.streams:
            ret["sardana"] = sardana_parse(event.streams["sardana"])

        if parameters["analysis_mode"].value == "roi":
            dat = None
            if "andor3_balor" in event.streams:
                dat = parse(event.streams["andor3_balor"])
            elif "andor3_zyla10" in event.streams:
                dat = parse(event.streams["andor3_zyla10"])
            elif "pilatus" in event.streams:
                dat = parse(event.streams["pilatus"])

            if dat:
                bg = parameters["background"].value
                if isinstance(dat, Stream1Start):
                    print("start message", dat)
                    return {**ret, "pileup_filename": dat.filename}
                if not isinstance(dat, Stream1Data):
                    return ret

                dark_corr = dat.data.clip(min=bg) - bg
                means = {}
                try:
                    rois = json.loads(parameters["rois"].value)
                    for rn, rd in rois.items():
                        tl = rd["handles"]["_handleBottomLeft"]
                        br = rd["handles"]["_handleTopRight"]

                        xslice = slice(min(int(tl[0]), int(br[0])), max(int(tl[0]), int(br[0])))
                        yslice = slice(min(int(tl[1]), int(br[1])), max(int(tl[1]), int(br[1])))

                        crop = dark_corr[yslice, xslice]
                        scalar = crop.mean()
                        means[rn] = scalar
                except:
                    pass

                return {"img": dark_corr , "cropped": None, "roi_means": means, **ret}

        elif parameters["analysis_mode"].data == b"sparsification":
            logger.debug("using parameters %s", parameters)
            bg = int(parameters["background"].data)
            thr = int(parameters["threshold"].data)
            size = int(parameters["spot_size"].data)
            dat = parse(event.streams["andor3_balor"])
            if not isinstance(dat, Stream1Data):
                return ret
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
            return {**ret, "spots": spots, "offsets": offsets}
        return ret

    def finish(self, parameters=None):
        print("finished")
