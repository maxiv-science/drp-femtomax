import asyncio
import glob
import json
import logging
import threading
import time

import h5pyd
import numpy as np
import pytest
from dranspose.replay import replay


def test_direct(tmp_path):
    roi_data = {
        "blub": {
            "visible": True,
            "type": "RectangleROI",
            "handles": {
                "_handleTopLeft": [0.0, 70.89219330855019],
                "_handleTopRight": [100.11524163568774, 100.89219330855019],
                "_handleBottomLeft": [0.0, 0.0],
                "_handleBottomRight": [81.11524163568774, 0.0],
                "_handleCenter": [40.55762081784387, 35.446096654275095],
                "_handleLabel": [0.0, 0.0],
            },
        }
    }

    params = [{"name": "rois", "data": json.dumps(roi_data)}]
    param_file = tmp_path / "params.json"
    with open(param_file, "w") as f:
        json.dump(params, f)

    replay(
        "src.worker:CmosWorker",
        "src.reducer:CmosReducer",
        glob.glob("data/balor*08c2eefb8b92.cbors"),
        None,
        param_file,
    )


def test_roi_values(tmp_path):
    stop_event = threading.Event()
    done_event = threading.Event()

    roi_data = {
        "blub": {
            "visible": True,
            "type": "RectangleROI",
            "handles": {
                "_handleTopLeft": [0.0, 70.89219330855019],
                "_handleTopRight": [100.11524163568774, 100.89219330855019],
                "_handleBottomLeft": [0.0, 0.0],
                "_handleBottomRight": [81.11524163568774, 0.0],
                "_handleCenter": [40.55762081784387, 35.446096654275095],
                "_handleLabel": [0.0, 0.0],
            },
        }
    }

    params = [{"name": "rois", "data": json.dumps(roi_data)}]
    param_file = tmp_path / "params.json"
    with open(param_file, "w") as f:
        json.dump(params, f)

    thread = threading.Thread(
        target=replay,
        args=(
            "src.worker:CmosWorker",
            "src.reducer:CmosReducer",
            glob.glob("data/balor*08c2eefb8b92.cbors"),
            None,
            param_file,
        ),
        kwargs={"port": 5010, "stop_event": stop_event, "done_event": done_event},
    )
    thread.start()

    # do live queries

    done_event.wait()

    f = h5pyd.File("http://localhost:5010/", "r")
    logging.info("file %s", list(f.keys()))
    logging.info("file %s", list(f["roi_means"].keys()))
    logging.info("file %s", list(f["step_means/blub/y"][:]))

    assert list(f["roi_means"].keys()) == ["blub"]
    assert np.array_equal(
        f["roi_means/blub"][:],
        [
            2.4387,
            0.0051,
            0.0027,
            0.0013,
            0.0029,
            0.0038,
            0.0025,
            0.0107,
            0.0061,
            0.0066,
            0.0066,
            0.0067,
            0.0144,
            0.004,
            0.0054,
            0.0079,
            0.0106,
            0.0099,
            0.0076,
            0.0068,
            0.0079,
            0.0134,
            0.0138,
            0.0124,
            0.0093,
        ],
    )
    assert np.array_equal(
        f["step_means/blub/y"][:],
        [
            0.49013999999999996,
            0.005940000000000001,
            0.00742,
            0.008560000000000002,
            0.01136,
        ],
    )
    assert dict(f["step_means/blub"].attrs.items()) == {
        "NX_class": "NXdata",
        "axes": ["x"],
        "signal": "y",
    }
    stop_event.set()

    thread.join()


@pytest.mark.skipif(
    "not config.getoption('dev')",
    reason="explicitly enable --dev elopment tests",
)
def test_keepalive(tmp_path):
    stop_event = threading.Event()
    done_event = threading.Event()

    roi_data = {
        "blub": {
            "visible": True,
            "type": "RectangleROI",
            "handles": {
                "_handleTopLeft": [0.0, 70.89219330855019],
                "_handleTopRight": [100.11524163568774, 100.89219330855019],
                "_handleBottomLeft": [0.0, 0.0],
                "_handleBottomRight": [81.11524163568774, 0.0],
                "_handleCenter": [40.55762081784387, 35.446096654275095],
                "_handleLabel": [0.0, 0.0],
            },
        }
    }

    params = [{"name": "rois", "data": json.dumps(roi_data)}]
    param_file = tmp_path / "params.json"
    with open(param_file, "w") as f:
        json.dump(params, f)

    thread = threading.Thread(
        target=replay,
        args=(
            "src.worker:CmosWorker",
            "src.reducer:CmosReducer",
            glob.glob("data/balor*08c2eefb8b92.cbors"),
            None,
            param_file,
        ),
        kwargs={"port": 5010, "stop_event": stop_event, "done_event": done_event},
    )
    thread.start()

    # do live queries

    done_event.wait()

    f = h5pyd.File("http://localhost:5010/", "r")
    logging.info("file %s", list(f.keys()))

    # assert f["accumulated_number"][()] == 21
    # assert f["image"][10, 10] == 21
    time.sleep(1000)
    stop_event.set()

    thread.join()
