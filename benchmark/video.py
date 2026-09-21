"""Streaming video output shared by the benchmark scripts."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def video_path(path: Path | None, count: int, multiple: bool) -> Path | None:
    if path is None or not multiple:
        return path
    return path.with_name(f"{path.stem}_{count}{path.suffix}")


@contextmanager
def video_writer(
    path: Path | None, fps: int, pixel_format: str = "rgb24"
) -> Iterator[Callable[[NDArray], None] | None]:
    if path is None:
        yield None
        return

    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 900
        stream.height = 600
        stream.pix_fmt = "yuv420p"

        def write(image: NDArray) -> None:
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format=pixel_format
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        try:
            yield write
        finally:
            for packet in stream.encode():
                container.mux(packet)
