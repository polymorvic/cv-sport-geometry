from pathlib import Path

import matplotlib as mpl
import pandas as pd
import tyro

from utils.const import SCENARIOS
from utils.court_finder import process
from utils.func import (
    create_reference_court,
    get_pictures,
    load_config,
    measure_error,
    plot_results,
    validate_data_and_pictures,
    warp_points,
)
from utils.schemas import TestConfig

mpl.rcParams["image.cmap"] = "gray"


def run(
    config_dir: Path = Path("config/test.config.json"),
    pics_dir: Path = Path("data/test"),
    results_df_name: str = "test_df",
) -> None:
    """
    Batch-detects and reconstructs tennis-court geometry for images in `pics_dir`
    using params from `config_dir`, and saves result plots under
    `{result_dir}/{TIMESTAMP_STRING}/`.

    For each image, calls `process(pic, param)` which:
      - extracts court lines (Canny+Hough), groups & intersects them,
      - uses `CourtFinder` to locate outer/inner baselines & netline, doubles/singles
        sidelines, centre service line, court centre, and service line,
      - returns dicts of Points and Lines and, if available, ground-truth points.
    Results are plotted with `plot_results`.

    Side effects:
      - creates the timestamped output directory and writes plots,
      - prints per-image errors; removes the run directory if it stays empty.

    Args:
        config_dir: Path to JSON config for image parameters.
        pics_dir: Directory with RGB input images.
        result_dir: Base output directory (run subdir named by `TIMESTAMP_STRING`).
    """
    config = load_config(config_dir, TestConfig)
    train_pics = get_pictures(pics_dir)["rgb"]
    test_df_rows = []

    validate_data_and_pictures(config.data, train_pics)

    for i, (data, train_pic) in enumerate(zip(config.data, train_pics, strict=False)):
        path = Path(config.testing_pics_dir) / config.run_name
        path.mkdir(exist_ok=True, parents=True)

        dst_points, dst_lines, ground_truth_points = (
            res if (res := process(train_pic, data)) is not None else (None, None, None)
        )

        if res is None:
            errors = dict.fromkeys(dst_points.keys(), None)
            scenario_errors = []
            row_dict = {"pic_index": i, "pic_name": data.pic_name, **errors}
            test_df_rows.append(row_dict)
            continue

        plot_results(train_pic, path, data.pic_name, dst_lines, dst_points)

        errors = measure_error(train_pic, dst_points, ground_truth_points)
        ref_points, ref_img = create_reference_court()

        scenario_errors = []
        for name, points in SCENARIOS:
            transformed_points, _, _ = warp_points(ref_points, dst_points, train_pic, ref_img, *points)
            error = measure_error(train_pic, transformed_points, ground_truth_points, name)
            scenario_errors.append(error)

        row_dict = {"pic_index": i, "pic_name": data.pic_name, **errors}

        for (scenario_name, _), err_dict in zip(SCENARIOS, scenario_errors, strict=False):
            if isinstance(err_dict, dict):
                # row_dict = {f"{scenario_name}_{k}": v for k, v in err_dict.items()}
                for k, v in err_dict.items():
                    row_dict[f"{scenario_name}_{k}"] = v

        test_df_rows.append(row_dict)

    df = pd.DataFrame.from_records(test_df_rows)
    base_cols = ["pic_index", "pic_name"] + list(errors.keys())
    scenario_cols = [c for c in df.columns if c not in base_cols]
    df[base_cols + scenario_cols].to_csv(path / f"{results_df_name}.csv", index=False)


if __name__ == "__main__":
    tyro.cli(run)
