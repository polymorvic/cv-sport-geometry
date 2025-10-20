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


def run(config_dir: Path = Path('config/run.config.json'), pics_dir: Path = Path('data/run'), result_dir: Path = Path("results/run")) -> None:
    """
    Detects and reconstructs tennis-court geometry from input images and saves result plots.
    
    Workflow (concise):
        1) Load image-processing params from `config/run.config.json` and RGB images from `data/run`.
        2) For each image:
           - Detect lines (Canny + Hough), build `Line` objects, filter verticals (`xv is None`), and group lines.
           - Compute all group–group intersections and initialize `CourtFinder`.
           - Find outer baseline/net points and sidelines; derive baseline and net line.
           - Scan for inner (singles) baselines and netline endpoints.
           - Derive centre service line, net service point, court centre, and service line.
           - Compose destination points/lines and plot/save results.
        3) On failure for an image, print the error; remove the (per-run) folder if it stayed empty.

    Side effects:
        - Creates the timestamped output directory; saves plots there.
        - Prints errors; deletes the output dir if no files were produced.

    Notes:
        Requires: TIMESTAMP_STRING, `ImageParams`, and helpers:
        `load_config`, `get_pictures`, `validate_data_and_pictures`,
        `apply_hough_transformation`, `Line`, `group_lines`, `CourtFinder`,
        `compose_court_data`, `plot_results`.

    Args:
        result_dir: Base output directory. Results are written to
            `{result_dir}/{TIMESTAMP_STRING}/`.
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
