from pathlib import Path

import tyro

from utils.const import TIMESTAMP_STRING
from utils.court_finder import process
from utils.func import (
    get_pictures,
    load_config,
    plot_results,
    validate_data_and_pictures,
)
from utils.schemas import ImageParams


def run(config_dir: Path = Path('config/run.config.json'), 
        pics_dir: Path = Path('data/run'), 
        result_dir: Path = Path("results/run")) -> None:
    """
    Runs batch tennis-court detection and reconstruction from images in `pics_dir`
    using parameters from `config_dir`, and saves plotted results under
    `{result_dir}/{TIMESTAMP_STRING}/`.

    For each image:
    - loads parameters and RGB data,
    - calls `process(pic, param)` to detect and reconstruct court geometry
        (lines, intersections, baselines, netlines, sidelines, service lines),
    - saves the resulting plots via `plot_results`.

    Side effects:
    - creates the timestamped output directory,
    - prints per-image errors,
    - removes the directory if no results are produced.

    Args:
        config_dir: Path to JSON config with image parameters.
        pics_dir: Directory containing input RGB images.
        result_dir: Base output directory for timestamped results.
    """
    
    path = result_dir / TIMESTAMP_STRING
    path.mkdir(exist_ok=True, parents=True)

    params = load_config(config_dir, ImageParams)
    pics = get_pictures(pics_dir)["rgb"]

    validate_data_and_pictures(params, pics)

    for pic, param in zip(pics, params, strict=False):

        dst_points, dst_lines, _ = (
            res if (res := process(pic, param)) is not None else (None, None, None)
        )

        plot_results(pic, path, param.pic_name, dst_lines, dst_points)

    if path.exists() and not any(path.iterdir()):
        path.rmdir()


if __name__ == "__main__":
    tyro.cli(run)
