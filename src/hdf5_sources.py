import itertools
import logging
import pickle
from typing import Generator

import numpy as np

from dranspose.event import InternalWorkerMessage, StreamData
from dranspose.protocol import EventNumber, StreamName
from dranspose.data.lecroy import (
    LecroyPrepare,
    LecroySeqEnd,
    LecroySeqStart,
    LecroyData,
    LecroyEnd,
    LECROY_TYPE,
)
import h5py
from bitshuffle import compress_lz4

logger = logging.getLogger(__name__)

MAX_EVENTS = 100


class LecroySource:  # Only works with old xes-receiver files
    def __init__(self) -> None:
        self.fname = "data/step-00000000.hdf5"
        self.fd = h5py.File(self.fname)
        self.traces = self.fd["/data/waveform1"]
        self.ts = self.fd["/data/timestamp1"]
        self.stream = StreamName("oscc_02_maui")

    def get_source_generators(
        self,
    ) -> list[Generator[InternalWorkerMessage, None, None]]:
        return [self.lecroy_source()]

    def lecroy_source(self) -> Generator[InternalWorkerMessage, None, None]:
        #
        # 1 b'{"htype": "msg", "what": 1, "frame": 0, "ntriggers": -1, "seqno": 0, "channels": [2, 4]}'
        lecroy_prep = (
            LecroyPrepare(htype="msg", what=0, frame=0).model_dump_json().encode()
        )
        start = InternalWorkerMessage(
            event_number=EventNumber(0),
            streams={self.stream: StreamData(typ=LECROY_TYPE, frames=[lecroy_prep])},
        )
        logger.debug(f"Sending {start=}")
        yield start

        # 1 b'{"htype": "msg", "what": 1, "frame": 0, "ntriggers": -1, "seqno": 0, "channels": [2, 4]}'
        lecroy_start = (
            LecroySeqStart(
                htype="msg", what=1, frame=0, ntriggers=-1, seqno=0, channels=[2]
            )
            .model_dump_json()
            .encode()
        )
        # 3 b'{"htype": "traces", "ch": 4, "ts": 1740563619.34, "frame": 33, "shape": [1, 8002], "horiz_offset": -1.0000966451309088e-07, "horiz_interval": 1.25000001668929e-11, "dtype": "float64"}'
        frameno = 0
        for trace, ts in zip(self.traces, self.ts):
            trace = np.expand_dims(trace, axis=0)
            meta = (
                LecroyData(
                    htype="traces",
                    ch=2,
                    ts=0,
                    frame=frameno,
                    shape=trace.shape,
                    horiz_offset=-1.0000966451309088e-07,
                    horiz_interval=1.25000001668929e-11,
                    dtype=str(trace.dtype),
                )
                .model_dump_json()
                .encode()
            )
            seq_end = (
                LecroySeqEnd(htype="msg", what=2, frame=frameno)
                .model_dump_json()
                .encode()
            )
            frames = [lecroy_start, meta, trace.tobytes(), pickle.dumps([ts]), seq_end]
            img = InternalWorkerMessage(
                event_number=EventNumber(frameno + 1),
                streams={self.stream: StreamData(typ=LECROY_TYPE, frames=frames)},
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
            streams={self.stream: StreamData(typ=LECROY_TYPE, frames=[lecroy_end])},
        )
        logger.debug(f"Sending {end=}")
        yield end


class LecroySequential:  # Only works with old xes-receiver files
    def __init__(self) -> None:
        self.fname = "data/step-00000000.hdf5"
        self.fd = h5py.File(self.fname)
        self.traces = self.fd["/data/waveform1"]
        self.ts = self.fd["/data/timestamp1"]
        self.stream = StreamName("oscc_02_maui")

    def get_source_generators(
        self,
    ) -> list[Generator[InternalWorkerMessage, None, None]]:
        return [self.lecroy_source()]

    def lecroy_source(self) -> Generator[InternalWorkerMessage, None, None]:
        #
        # 1 b'{"htype": "msg", "what": 1, "frame": 0, "ntriggers": -1, "seqno": 0, "channels": [2, 4]}'
        lecroy_prep = (
            LecroyPrepare(htype="msg", what=0, frame=0).model_dump_json().encode()
        )
        start = InternalWorkerMessage(
            event_number=EventNumber(0),
            streams={self.stream: StreamData(typ=LECROY_TYPE, frames=[lecroy_prep])},
        )
        logger.debug(f"Sending {start=}")
        yield start

        # 1 b'{"htype": "msg", "what": 1, "frame": 0, "ntriggers": -1, "seqno": 0, "channels": [2, 4]}'
        lecroy_start = (
            LecroySeqStart(
                htype="msg", what=1, frame=0, ntriggers=-1, seqno=0, channels=[2]
            )
            .model_dump_json()
            .encode()
        )
        # 3 b'{"htype": "traces", "ch": 4, "ts": 1740563619.34, "frame": 33, "shape": [1, 8002], "horiz_offset": -1.0000966451309088e-07, "horiz_interval": 1.25000001668929e-11, "dtype": "float64"}'
        frameno = 0
        trace_list = []
        ts_list = []
        for trace, ts in zip(self.traces, self.ts):
            trace_list.append(trace)
            ts_list.append([ts])
            if len(trace_list) % 20 == 0:
                traces = np.array(trace_list)
                meta = (
                    LecroyData(
                        htype="traces",
                        ch=2,
                        ts=ts_list[0][0],
                        frame=frameno,
                        shape=traces.shape,
                        horiz_offset=-1.0000966451309088e-07,
                        horiz_interval=1.25000001668929e-11,
                        dtype=str(traces.dtype),
                    )
                    .model_dump_json()
                    .encode()
                )
                seq_end = (
                    LecroySeqEnd(htype="msg", what=2, frame=frameno)
                    .model_dump_json()
                    .encode()
                )
                frames = [
                    lecroy_start,
                    meta,
                    traces.tobytes(),
                    pickle.dumps(ts_list),
                    seq_end,
                ]
                img = InternalWorkerMessage(
                    event_number=EventNumber(frameno + 1),
                    streams={self.stream: StreamData(typ=LECROY_TYPE, frames=frames)},
                )
                yield img
                trace_list = []
                ts_list = []
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
            streams={self.stream: StreamData(typ=LECROY_TYPE, frames=[lecroy_end])},
        )
        logger.debug(f"Sending {end=}")
        yield end
