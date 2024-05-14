import logging
import tempfile
import json

from dranspose.event import EventData
from dranspose.parameters import IntParameter, StrParameter, BoolParameter
from dranspose.middlewares.stream1 import parse
from dranspose.middlewares.sardana import parse as sardana_parse
from dranspose.data.stream1 import Stream1Data, Stream1Start, Stream1End
import numpy as np
from numpy import unravel_index
from scipy.ndimage import gaussian_filter, center_of_mass

logger = logging.getLogger(__name__)

def hithist(img, threshold_counting, pre_threshold): # This is the magic function that counts the photons using a 3x3 square (you can change the size of the box)
    fullframe = img.clip(min=threshold_counting) - threshold_counting
    positions = np.asarray(fullframe > 0).nonzero()
    hits = set()
    for x,y in zip(*positions):
        small = fullframe[x-1:x+2, y-1:y+2]
        if np.prod(small.shape) == 0:
            continue

        maxpos = np.unravel_index(small.argmax(), small.shape)
        hits.add( (x+maxpos[0]-1, y+maxpos[1]-1))
    pos = []
    for x,y in hits:
        hit = fullframe[x-1:x+2,y-1:y+2]
        d1 = hit.sum()
        cm = center_of_mass(hit)
        pos.append((x+cm[0]-1, y+cm[1]-1, d1))
    return [p for p in pos if p[2] > pre_threshold]

class CmosWorker:

    @staticmethod
    def describe_parameters():
        params = [
            IntParameter(name="background", default=0),
            IntParameter(name="threshold", default=12),
            IntParameter(name="sigma", default=4),
            IntParameter(name="spot_size", default=7),
            BoolParameter(name="blur", default=False),
            StrParameter(name="analysis_mode", default="roi"),
            IntParameter(name="threshold_counting", default=108), # important threshold for counting photons later
            IntParameter(name="pre_threshold", default=5), # discard every spot with a total energy below that
            StrParameter(name="rois", default="{}"),
        ]
        return params

    def __init__(self, **kwargs):
        self.number = 0
        self.cumu = None

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
            dat = None
            if "andor3_balor" in event.streams:
                dat = parse(event.streams["andor3_balor"])
            elif "andor3_zyla10" in event.streams:
                dat = parse(event.streams["andor3_zyla10"])

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

        elif parameters["analysis_mode"].value == "cog":
            print("bla")
            logger.debug("cog using parameters %s", parameters)
            threshold_counting = parameters["threshold_counting"].value
            pre_threshold = parameters["pre_threshold"].value
            print("vals", threshold_counting, pre_threshold)
            dat = None
            if "andor3_balor" in event.streams:
                dat = parse(event.streams["andor3_balor"])
            elif "andor3_zyla10" in event.streams:
                dat = parse(event.streams["andor3_zyla10"])

            print(dat)
            if isinstance(dat, Stream1Start):
                print("using filename", dat.filename)
                return {**ret, "cog_filename": dat.filename}

            if isinstance(dat, Stream1End):
                print("send off accumulated image")
                return {**ret, "reconstructed": self.cumu}

            if not isinstance(dat, Stream1Data):
                return ret

            print("there is data")

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
                        self.cumu = np.zeros_like(dat.data)[yslice,xslice]

                    for hit in hits:
                        self.cumu[int(hit[0]), int(hit[1])] += 1
                    #print(hits)
            except:
                pass

            return {**ret, "hits": hits, "frame": dat.frame}
        return ret

    def finish(self, parameters=None):
        print("finished")
