import itertools
import logging
from typing import Generator

from dranspose.event import InternalWorkerMessage, StreamData
from dranspose.protocol import EventNumber, StreamName
from dranspose.data.lecroy import LecroyStart, LecroyData, LecroyEnd
import h5py
from bitshuffle import compress_lz4

logger = logging.getLogger(__name__)

MAX_EVENTS = 100


class LecroySource:  # Only works with old xes-receiver files
    def __init__(self) -> None:
        self.fname = "data/step-00000000.hdf5"
        self.fd = h5py.File(self.fname)
        self.dset = self.fd["/data/waveform1"]

    def get_source_generators(
        self,
    ) -> list[Generator[InternalWorkerMessage, None, None]]:
        return [self.lecroy_source()]

    def lecroy_source(self) -> Generator[InternalWorkerMessage, None, None]:
        # msg_number = itertools.count(0)
        #
        # 1 b'{"htype": "msg", "what": 1, "frame": 0, "ntriggers": -1, "seqno": 0, "channels": [2, 4]}'
        lecroy_start = (
            LecroyStart(
                htype="msg", what=1, frame=0, ntriggers=-1, seqno=0, channels=[2]
            )
            .model_dump_json()
            .encode()
        )
        start = InternalWorkerMessage(
            event_number=EventNumber(0),
            streams={
                StreamName("oscc-02-seq-maui"): StreamData(
                    typ="lecroy", frames=[lecroy_start]
                )
            },
        )
        logger.debug(f"Sending {start=}")
        yield start

        # 3 b'{"htype": "traces", "ch": 4, "ts": 1740563619.34, "frame": 33, "shape": [1, 8002], "horiz_offset": -1.0000966451309088e-07, "horiz_interval": 1.25000001668929e-11, "dtype": "float64"}'
        frameno = 0
        for trace in self.dset:
            meta = (
                LecroyData(
                    htype="traces",
                    ch=2,
                    ts=0,
                    frame=frameno,
                    shape=trace.shape,
                    horiz_offset=0,
                    horiz_interval=0,
                    dtype=str(trace.dtype),
                )
                .model_dump_json()
                .encode()
            )
            img = InternalWorkerMessage(
                event_number=EventNumber(frameno + 1),
                streams={
                    StreamName("oscc-02-seq-maui"): StreamData(
                        typ="lecroy", frames=[meta, trace.tobytes()]
                    )
                },
            )
            yield img
            frameno += 1
            # logger.debug(f"Sending {img=}")
            if frameno > MAX_EVENTS:
                break

        # 1 b'{"htype": "msg", "what": 3, "frame": 66, "frames": 66}'
        lecroy_end = (
            LecroyEnd(htype="msg", what=3, frame=frameno, frames=frameno)
            .model_dump_json()
            .encode()
        )
        end = InternalWorkerMessage(
            event_number=EventNumber(frameno),
            streams={
                StreamName("oscc-02-seq-maui"): StreamData(
                    typ="lecroy", frames=[lecroy_end]
                )
            },
        )
        logger.debug(f"Sending {end=}")
        yield end
