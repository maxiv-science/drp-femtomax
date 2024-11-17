import itertools
import logging
import time
from typing import Generator

import h5py
import numpy as np
import pickle

from dranspose.event import InternalWorkerMessage, StreamData
from dranspose.protocol import EventNumber, StreamName
from dranspose.data.stream1 import Stream1Start, Stream1Data, Stream1End


logger = logging.getLogger(__name__)


class BalorSource:  # Only works with old xes-receiver files
    def __init__(self) -> None:
        pass

    def get_source_generators(
        self,
    ) -> list[Generator[InternalWorkerMessage, None, None]]:
        return [self.balor_source()]

    def balor_source(self) -> Generator[InternalWorkerMessage, None, None]:
        msg_number = itertools.count(0)

        stins_start = (
            Stream1Start(
                htype="header",
                filename="balor-photons.h5",
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

        with h5py.File("data/scan-103327_small.h5") as fh:
            frameno = 0
            dset = fh["/entry/instrument/balor/data"]
            for image_idx in range(10):
                image = dset[image_idx%dset.shape[0]][:]
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

