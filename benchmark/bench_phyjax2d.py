import csv
from datetime import datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer

from phyjax2d import SpaceBuilder, Vec2d, nstep, step
from phyjax2d.moderngl_vis import MglVisualizer

if __package__:
    from .video import video_path, video_writer
else:
    from video import video_path, video_writer


def ball_fall_phyjax2d(
    n_balls: int,
    debug_vis: bool,
    n_iter: int = 1000,
    videopath: Path | None = None,
) -> timedelta:
    """
    Simulates n_balls falling using phyjax2d.
    Render to a window with debug_vis, or save frames with videopath.
    """
    builder = SpaceBuilder(
        gravity=(0.0, -900.0),
        dt=0.01,
        viscous_damping=0.6,
        n_velocity_iter=4,
        n_position_iter=1,
        bias_factor=0.1,
        bounce_threshold=4,
        allowed_penetration=0.1,
    )

    for _ in range(n_balls):
        builder.add_circle(
            radius=4.0,
            density=1.0 / (16 * np.pi),
            elasticity=0.5,
            friction=0.5,
        )

    # Container setup
    builder.add_segment(
        p1=Vec2d(50.0, 50.0),
        p2=Vec2d(850.0, 50.0),
        elasticity=0.4,
        friction=0.5,
    )
    builder.add_segment(
        p1=Vec2d(50.0, 50.0),
        p2=Vec2d(50.0, 550.0),
        elasticity=0.4,
        friction=0.5,
    )
    builder.add_segment(
        p1=Vec2d(850.0, 50.0),
        p2=Vec2d(850.0, 550.0),
        elasticity=0.4,
        friction=0.5,
    )

    space = builder.build()

    # 2. Initialize State
    rng = np.random.default_rng()
    x_coords = rng.uniform(100, 800, n_balls)
    y_coords = rng.uniform(150, 500, n_balls)
    pos_array = jnp.stack([jnp.array(x_coords), jnp.array(y_coords)], axis=-1)

    sd = space.zeros_state().nested_replace("circle.p.xy", pos_array)
    vs = space.init_solver()

    # 3. Initialize Visualizer
    if debug_vis or videopath is not None:
        # We define the range based on the window size/container
        visualizer = MglVisualizer(
            x_range=900.0,
            y_range=600.0,
            space=space,
            stated=sd,
            title=f"Phyjax2D Debug: {n_balls} balls",
            figsize=(900, 600),
            backend="pyglet" if debug_vis else "headless",
        )
        jit_step = jax.jit(step, static_argnums=(0,))
        start = datetime.now()
        try:
            with video_writer(videopath, fps=100, pixel_format="rgba") as write_frame:
                for _ in range(n_iter):
                    sd, _, _ = jit_step(space, sd, vs)
                    visualizer.render(state=sd)
                    if write_frame is not None:
                        write_frame(visualizer.get_image())
                    if debug_vis:
                        visualizer.show()
        finally:
            visualizer.close()
        return datetime.now() - start
    else:

        @jax.jit
        def n_step_fixed(sd, vs):
            sd, vs, _ = nstep(5, 0.6, space, sd, vs)
            return sd, vs.replace(pn=vs.pn * 0.6)

        n_step_fixed(sd, vs)

        start = datetime.now()
        for _ in range(n_iter // 5):
            sd, vs = n_step_fixed(sd, vs)
        return datetime.now() - start


DEFAULT_COUNTS = [1000]


def main(
    counts: list[int] = DEFAULT_COUNTS,
    debug_vis: bool = False,
    n_iter: int = 1000,
    filename: Path = Path("bench.csv"),
    videopath: Path | None = None,
) -> None:
    results = []

    for count in counts:
        duration = ball_fall_phyjax2d(
            count,
            debug_vis,
            n_iter=n_iter,
            videopath=video_path(videopath, count, len(counts) > 1),
        )
        # Convert timedelta to total seconds as a float for the CSV
        seconds = duration.total_seconds()
        results.append((count, seconds))

    if not debug_vis:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["n_balls", "duration_seconds"])  # Header
            writer.writerows(results)


if __name__ == "__main__":
    typer.run(main)
