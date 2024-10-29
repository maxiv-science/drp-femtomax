import logging
import threading
from glob import glob

import h5pyd
from dranspose.replay import replay


def test_replay_gen():
    replay(
        "src.worker:CmosWorker",
        "src.reducer:CmosReducer",
        None,
        "src.saved_source:BalorSource",
        "testparams.json",
    )
