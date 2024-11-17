import itertools
import logging
import time
from typing import Generator
import numpy as np
import pickle

from dranspose.event import InternalWorkerMessage, StreamData
from dranspose.protocol import EventNumber, StreamName
from dranspose.data.stream1 import Stream1Start, Stream1Data, Stream1End


logger = logging.getLogger(__name__)


class BalorSource:  # Only works with old xes-receiver files
    def __init__(self) -> None:
        self.image = np.ones((4104, 4128), dtype=np.uint16) * 113
        # np.load("data/test-images.npz")["arr_0"]

    def get_source_generators(
        self,
    ) -> list[Generator[InternalWorkerMessage, None, None]]:
        return [self.balor_source(), self.sardana_file_source()]

    def balor_source(self) -> Generator[InternalWorkerMessage, None, None]:
        msg_number = itertools.count(0)

        stins_start = (
            Stream1Start(
                htype="header",
                filename="automated-test.h5",
                msg_number=next(msg_number),
            )
            .model_dump_json()
            .encode()
        )
        start = InternalWorkerMessage(
            event_number=EventNumber(0),
            streams={
                StreamName("andor3_balor"): StreamData(
                    typ="STINS", frames=[stins_start]
                )
            },
        )
        logger.debug(f"Sending {start=}")
        yield start

        frameno = 0
        for image in [self.image] * 21:
            stins = (
                Stream1Data(
                    htype="image",
                    msg_number=next(msg_number),
                    frame=frameno,
                    shape=image.shape,
                    compression="none",
                    type=str(image.dtype),
                )
                .model_dump_json()
                .encode()
            )
            dat = image
            img = InternalWorkerMessage(
                event_number=EventNumber(frameno + 1),
                streams={
                    StreamName("andor3_balor"): StreamData(
                        typ="STINS", frames=[stins, dat.tobytes()]
                    )
                },
            )
            yield img
            frameno += 1
            time.sleep(0.1)
            # logger.debug(f"Sending {img=}")

        stins_end = (
            Stream1End(htype="series_end", msg_number=next(msg_number))
            .model_dump_json()
            .encode()
        )
        end = InternalWorkerMessage(
            event_number=EventNumber(frameno),
            streams={
                StreamName("andor3_balor"): StreamData(typ="STINS", frames=[stins_end])
            },
        )
        logger.debug(f"Sending {end=}")
        yield end

    def sardana_file_source(self):
        with open(
            "data/fullimgsardana-ingester-cf9c9d60-6509-4ce3-a721-f01b4d328d9f.pkls",
            "rb",
        ) as f:
            while True:
                try:
                    msg = pickle.load(f)
                    assert isinstance(msg, InternalWorkerMessage)
                    yield msg
                except EOFError:
                    break
