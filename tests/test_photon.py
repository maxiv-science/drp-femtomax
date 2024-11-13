import asyncio
import json
import logging
import threading
import time

import h5pyd
from dranspose.replay import replay

def test_direct(tmp_path):

    roi_data = {'photon': {'visible': True,
          'type': 'RectangleROI',
          'handles': {'_handleTopLeft': [0.0, 70.89219330855019],
           '_handleTopRight': [2001.11524163568774, 3000.89219330855019],
           '_handleBottomLeft': [1500.0, 1500.0],
           '_handleBottomRight': [81.11524163568774, 0.0],
           '_handleCenter': [40.55762081784387, 35.446096654275095],
           '_handleLabel': [0.0, 0.0]}}
            }

    params = [{"name": "rois", "data": json.dumps(roi_data)}]
    param_file = tmp_path / "params.json"
    with open(param_file, "w") as f:
        json.dump(params, f)

    replay("src.worker:CmosWorker",
        "src.reducer:CmosReducer",
        None,
        "src.photon_source:BalorSource",
        param_file)

def est_non_photon(tmp_path):


    params = [{"name": "cmos_background", "data": "100"},
              {"name": "cmos_threshold", "data": "15"}]
    param_file = tmp_path / "params.json"
    with open(param_file, "w") as f:
        json.dump(params, f)

    replay("src.worker:CmosWorker",
        "src.reducer:CmosReducer",
        None,
        "src.photon_source:BalorSource",
        param_file)

def est_pipeline():
    stop_event = threading.Event()
    done_event = threading.Event()

    thread = threading.Thread(
        target=replay,
        args=(
            "src.worker:CmosWorker",
            "src.reducer:CmosReducer",
            None,
            "src.photon_source:BalorSource",
            "testparams.json",
        ),
        kwargs={"port": 5010, "stop_event": stop_event, "done_event": done_event},
    )
    thread.start()

    # do live queries

    done_event.wait()

    f = h5pyd.File("http://localhost:5010/", "r")
    logging.info("file %s", list(f.keys()))

    assert f["accumulated_number"][()] == 21
    assert f["image"][10, 10] == 21

    stop_event.set()

    thread.join()
