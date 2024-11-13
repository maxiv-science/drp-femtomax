import asyncio
import logging
import threading
import time

import h5pyd
from dranspose.replay import replay


def test_pipeline():
    stop_event = threading.Event()
    done_event = threading.Event()

    thread = threading.Thread(
        target=replay,
        args=(
            "src.worker:CmosWorker",
            "src.reducer:CmosReducer",
            None,
            "src.saved_source:BalorSource",
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
