# phyjax2d

A jax-based 2d physics library, mainly intended to use for reinforcement learning research.

# License
[Apache LICENSE 2.0](./LICENSE) holds unless otherwise noted.

`vec2d.py` is copied from [PyMunk](pymunk.org) with the license-header as-is.

### Benchmark videos

Install development and visualization dependencies with `uv sync --extra vis`, then
record a benchmark without opening a window:

```sh
uv run python benchmark/bench_phyjax2d.py --counts 100 --n-iter 1000 --videopath simulation.mp4
```

`--videopath` is also supported by `bench_phyjax2d_xpbd.py` and `bench_pymunk.py`.
Add `--debug-vis` to display the simulation while recording. Videos use H.264 at
900 × 600, with one frame per simulation step and playback matching simulated time.
For multiple `--counts`, filenames include the count (for example,
`simulation_100.mp4`). Recording timings include rendering and encoding overhead.
