import json

import numpy as np


def parse_rois(parameters) -> dict[str, tuple[slice, slice]]:
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