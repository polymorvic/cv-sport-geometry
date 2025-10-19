from datetime import datetime
from pathlib import Path

import tyro

from src.utils.const import TIMESTAMPT_STRING
from src.utils.corners import CourtFinder
from src.utils.func import (
    apply_hough_transformation,
    compose_court_data,
    get_pictures,
    group_lines,
    load_config,
    plot_results,
    validate_data_and_pictures,
)
from src.utils.lines import Line
from src.utils.schemas import ImageParams


def run(result_dir: Path = Path("results/run")) -> None:
    path = result_dir / TIMESTAMPT_STRING
    path.mkdir(exist_ok=True, parents=True)

    params = load_config("config/run.config.json", ImageParams)
    pics = get_pictures("data/run")["rgb"]

    validate_data_and_pictures(params, pics)

    for pic, param in zip(pics, params, strict=False):
        pic_hough_lines, line_endpoints = apply_hough_transformation(pic)
        line_objs = [Line.from_hough_line(line[0]) for line in line_endpoints]
        line_objs = [line for line in line_objs if line.xv is None]

        grouped_lines = group_lines(line_objs)

        pic_line_intersections = pic.copy()
        intersections = []
        for group1 in grouped_lines:
            for group2 in grouped_lines:
                intersection = group1.intersection(group2, pic_line_intersections)
                if intersection is not None:
                    intersections.append(intersection)

        candidates = CourtFinder(intersections, pic)

        try:
            closer_outer_baseline_intersecetion, closer_outer_netintersection, used_line = (
                candidates.find_closer_outer_baseline_point()
            )
            closer_outer_baseline_point = closer_outer_baseline_intersecetion.point
            closer_outer_netline_point = closer_outer_netintersection.point

            closer_outer_sideline = Line.from_points(closer_outer_baseline_point, closer_outer_netline_point)

            further_outer_baseline_intersection, last_local_line = candidates.find_further_outer_baseline_intersection(
                closer_outer_baseline_intersecetion,
                used_line,
                param.canny_thresh.lower,
                param.canny_thresh.upper,
                offset=param.offset,
            )
            further_outer_baseline_point = further_outer_baseline_intersection.point

            baseline = Line.from_points(closer_outer_baseline_point, further_outer_baseline_point)

            netline = candidates.find_netline(closer_outer_netline_point, baseline, param.max_line_gap)

            further_outer_sideline = candidates.find_further_doubles_sideline(
                further_outer_baseline_point,
                last_local_line,
                param.offset,
                param.extra_offset,
                param.bin_thresh,
                param.surface_type,
            )
            further_outer_netline_point = further_outer_sideline.intersection(netline, pic).point

            closer_inner_baseline_point, further_inner_baseline_point = candidates.scan_endline(
                baseline,
                netline,
                closer_outer_baseline_point,
                further_outer_netline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                param.bin_thresh_endline_scan.baseline,
                param.canny_thresh.lower,
                param.canny_thresh.upper,
                param.max_line_gap,
                searching_line="base",
            )

            closer_inner_netline_point, further_inner_netline_point = candidates.scan_endline(
                baseline,
                netline,
                closer_outer_baseline_point,
                further_outer_netline_point,
                closer_outer_netline_point,
                further_outer_baseline_point,
                param.bin_thresh_endline_scan.netline,
                param.canny_thresh.lower,
                param.canny_thresh.upper,
                param.max_line_gap,
                searching_line="net",
            )

            closer_inner_sideline = Line.from_points(closer_inner_baseline_point, closer_inner_netline_point)
            further_inner_sideline = Line.from_points(further_inner_baseline_point, further_inner_netline_point)

            net_service_point, centre_service_line = candidates.find_net_service_point_centre_service_line(
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
                param.bin_thresh_centre_service_line,
                param.canny_thresh.lower,
                param.canny_thresh.upper,
                param.max_line_gap_centre_service_line,
                param.min_line_len_ratio,
                param.hough_thresh,
            )

            centre_service_point, further_service_point, closer_service_point = candidates.find_center(
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
                param.bin_thresh_centre_service_line,
                param.canny_thresh.lower,
                param.canny_thresh.upper,
                param.max_line_gap_centre_service_line,
                param.min_line_len_ratio,
                param.hough_thresh,
            )

            service_line = Line.from_points(further_service_point, closer_service_point)

            dst_points, dst_lines = compose_court_data(
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
            )

            plot_results(pic, path, param.pic_name, dst_lines, dst_points)

        except Exception as e:
            print(e)

        finally:
            if path.exists() and not any(path.iterdir()):
                path.rmdir()


if __name__ == "__main__":
    tyro.cli(run)
