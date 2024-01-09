import json
import pickle
import random

import h5py
import numpy as np
import zmq
from h5py import Dataset

from dranspose.event import StreamData, EventData, ResultData
from dranspose.data.stream1 import Stream1Data
from dranspose.protocol import WorkParameter

from worker import CmosWorker
from reducer import CmosReducer


parameters={"background": WorkParameter(name="background", data="100"),
            "threshold": WorkParameter(name="threshold", data="20"),
            "spot_size": WorkParameter(name="spot_size", data="7"),
            "filename": WorkParameter(name="filename", data="output.h5")}

images = np.load("../test-images.npz")["arr_0"]


workers = [CmosWorker(parameters=parameters) for  _ in range(2)]
reducer = CmosReducer(parameters=parameters)

for i, image in enumerate(images):

    img_header = Stream1Data(msg_number=i+1, htype="image", frame=i, shape=image.shape, type="uint16", compression="none")
    img_header.model_dump_json()
    balor = StreamData(typ="STINS", frames=[img_header.model_dump_json(), image.tobytes()])
    ev = EventData(event_number=i, streams={"balor": balor})

    wi = random.randint(0, len(workers) - 1)
    data = workers[wi].process_event(ev, parameters=parameters)
    rd = ResultData(
        event_number=i,
        worker=f"worker{wi}",
        payload=data,
        parameters_hash="asd",
    )

    reducer.process_result(rd, parameters=parameters)

[worker.finish(parameters=parameters) for worker in workers]
reducer.finish(parameters=parameters)