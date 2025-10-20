from pathlib import Path

import matplotlib as mpl
import pandas as pd
import tyro

from utils.court_finder import CourtFinder
from utils.func import (
    apply_hough_transformation,
    compose_court_data,
    create_reference_court,
    get_pictures,
    group_lines,
    load_config,
    measure_error,
    plot_results,
    validate_data_and_pictures,
    warp_points,
)
from utils.lines import Line
from utils.schemas import TestConfig

mpl.rcParams["image.cmap"] = "gray"

def run(config_dir: Path = Path('config/test.config.json'), pics_dir: Path = Path('data/test'), results_df_name: str = 'test_df') -> None:
    config = load_config(config_dir, TestConfig)
    train_pics = get_pictures(pics_dir)["rgb"]
    test_df_rows = []

    validate_data_and_pictures(config.data, train_pics)

    for i, (data, train_pic) in enumerate(zip(config.data, train_pics, strict=False)):
        path = Path(config.testing_pics_dir) / config.run_name
        path.mkdir(exist_ok=True, parents=True)

        train_pic_hough_lines, line_endpoints = apply_hough_transformation(train_pic)
        line_objs = [Line.from_hough_line(line[0]) for line in line_endpoints]
        line_objs = [line for line in line_objs if line.xv is None]

        grouped_lines = group_lines(line_objs)

        train_pic_line_intersections = train_pic.copy()
        intersections = []
        for group1 in grouped_lines:
            for group2 in grouped_lines:
                intersection = group1.intersection(group2, train_pic_line_intersections)
                if intersection is not None:
                    intersections.append(intersection)

        court_finder = CourtFinder(intersections, train_pic)

        try:
            closer_outer_baseline_intersecetion, closer_outer_netintersection, used_line = (
                court_finder.find_closer_outer_baseline_point()
            )
            closer_outer_baseline_point = closer_outer_baseline_intersecetion.point
            closer_outer_netline_point = closer_outer_netintersection.point

            closer_outer_sideline = Line.from_points(closer_outer_baseline_point, closer_outer_netline_point)

            further_outer_baseline_intersection, last_local_line = court_finder.find_further_outer_baseline_intersection(
                closer_outer_baseline_intersecetion,
                used_line,
                data.canny_thresh.lower,
                data.canny_thresh.upper,
                offset=data.offset,
            )
            further_outer_baseline_point = further_outer_baseline_intersection.point

            baseline = Line.from_points(closer_outer_baseline_point, further_outer_baseline_point)
            netline = court_finder.find_netline(closer_outer_netline_point, baseline, data.max_line_gap)

            further_outer_sideline = court_finder.find_further_doubles_sideline(
                further_outer_baseline_point,
                last_local_line,
                data.offset,
                data.extra_offset,
                data.bin_thresh,
                data.surface_type,
            )
            further_outer_netline_point = further_outer_sideline.intersection(netline, train_pic).point

            closer_inner_baseline_point, further_inner_baseline_point = court_finder.scan_endline(
                baseline,
                netline,
                closer_outer_baseline_point,
                further_outer_netline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                data.bin_thresh_endline_scan.baseline,
                data.canny_thresh.lower,
                data.canny_thresh.upper,
                data.max_line_gap,
                searching_line="base",
            )

            closer_inner_netline_point, further_inner_netline_point = court_finder.scan_endline(
                baseline,
                netline,
                closer_outer_baseline_point,
                further_outer_netline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                data.bin_thresh_endline_scan.netline,
                data.canny_thresh.lower,
                data.canny_thresh.upper,
                data.max_line_gap,
                searching_line="net",
            )

            closer_inner_sideline = Line.from_points(closer_inner_baseline_point, closer_inner_netline_point)
            further_inner_sideline = Line.from_points(further_inner_baseline_point, further_inner_netline_point)

            net_service_point, centre_service_line = court_finder.find_net_service_point_centre_service_line(
                closer_outer_baseline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                further_outer_netline_point,
                closer_inner_baseline_point,
                further_inner_baseline_point,
                closer_inner_netline_point,
                further_inner_netline_point,
                baseline,
                netline,
                data.bin_thresh_centre_service_line,
                data.canny_thresh.lower,
                data.canny_thresh.upper,
                data.max_line_gap_centre_service_line,
                data.min_line_len_ratio,
                data.hough_thresh,
            )

            centre_service_point, further_service_point, closer_service_point = court_finder.find_center(
                closer_outer_baseline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                further_outer_netline_point,
                closer_inner_baseline_point,
                further_inner_baseline_point,
                closer_inner_netline_point,
                further_inner_netline_point,
                baseline,
                closer_inner_sideline,
                further_inner_sideline,
                centre_service_line,
                data.bin_thresh_centre_service_line,
                data.canny_thresh.lower,
                data.canny_thresh.upper,
                data.max_line_gap_centre_service_line,
                data.min_line_len_ratio,
                data.hough_thresh,
            )

            service_line = Line.from_points(further_service_point, closer_service_point)

            dst_points, dst_lines, ground_truth_points = compose_court_data(
                closer_outer_baseline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                further_outer_netline_point,
                closer_inner_baseline_point,
                further_inner_baseline_point,
                closer_inner_netline_point,
                further_inner_netline_point,
                net_service_point,
                centre_service_point,
                further_service_point,
                closer_service_point,
                closer_outer_sideline,
                baseline,
                netline,
                further_outer_sideline,
                closer_inner_sideline,
                further_inner_sideline,
                centre_service_line,
                service_line,
                data,
            )

            plot_results(train_pic, path, data.pic_name, dst_lines, dst_points)

            errors = measure_error(train_pic, dst_points, ground_truth_points)
            ref_points, ref_img = create_reference_court()

            scenario_errors = []
            scenarios = [
                (
                    "test1",
                    [
                        "closer_outer_baseline_point",
                        "closer_outer_netline_point",
                        "closer_inner_netline_point",
                        "closer_inner_baseline_point",
                    ],
                ),
                (
                    "test2",
                    [
                        "closer_outer_baseline_point",
                        "closer_inner_baseline_point",
                        "further_inner_baseline_point",
                        "further_outer_baseline_point",
                        "further_outer_netline_point",
                        "net_service_point",
                        "closer_inner_netline_point",
                        "closer_outer_netline_point",
                    ],
                ),
                (
                    "test3",
                    [
                        "closer_outer_baseline_point",
                        "closer_inner_baseline_point",
                        "closer_service_point",
                        "centre_service_point",
                        "further_service_point",
                    ],
                ),
                (
                    "test4",
                    [
                        "further_outer_baseline_point",
                        "further_inner_baseline_point",
                        "further_service_point",
                        "centre_service_point",
                        "net_service_point",
                    ],
                ),
            ]

            for name, points in scenarios:
                transformed_points, _, _ = warp_points(ref_points, dst_points, train_pic, ref_img, *points)
                error = measure_error(train_pic, transformed_points, ground_truth_points, name)
                scenario_errors.append(error)

        except Exception as e:
            print(e)
            errors = dict.fromkeys(dst_points.keys(), None)
            scenario_errors = []

        finally:
            row_dict = {"pic_index": i, "pic_name": data.pic_name, **errors}

        for (scenario_name, _), err_dict in zip(scenarios, scenario_errors, strict=False):
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
