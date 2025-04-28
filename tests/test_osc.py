import logging
import threading
import pytest

import h5pyd
from dranspose.replay import replay


# @pytest.mark.parametrize("src", [("src.hdf5_sources:LecroySource", "src.hdf5_sources:LecroySequential")])
def test_pipeline():
    stop_event = threading.Event()
    done_event = threading.Event()

    thread = threading.Thread(
        target=replay,
        args=(
            "src.worker:CmosWorker",  # -w WORKERCLASS; see dranspose replay -h
            "src.reducer:CmosReducer",  # -r REDUCERCLASS; see dranspose replay -h
            None,  # -f FILES [FILES ...]; see dranspose replay -h
            "src.hdf5_sources:LecroySequential",  # -s SOURCE; see dranspose replay -h
            "testparams.json",  # -p PARAMETERS; see dranspose replay -h
        ),
        kwargs={"port": 5010, "stop_event": stop_event, "done_event": done_event},
    )
    thread.start()

    # do live queries
    done_event.wait()

    f = h5pyd.File("http://localhost:5010/", "r")
    logging.info("file %s", list(f.keys()))

    assert dict(list(f["osc"]["channel_02"]["hist"].attrs.items())) == {
        "NX_class": "NXdata",
        "axes": ["x"],
        "signal": "y",
    }
    assert len(f["osc"]["channel_02"]["pos"]) == 165
    assert len(f["osc"]["channel_02"]["amps"]) == 165
    assert len(f["osc"]["channel_02"]["hist"]["y"]) == 100

    stop_event.set()

    thread.join()
