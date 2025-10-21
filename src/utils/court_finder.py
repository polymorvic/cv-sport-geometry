from collections import defaultdict
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from .const import SETTINGS
from .func import (
    _plot_objs,
    apply_hough_transformation,
    check_items_sign,
    compose_court_data,
    crop_court_field,
    detect_lines_opposite_slope,
    draw_and_display,
    find_net_lines,
    find_point_neighbourhood,
    find_point_neighbourhood_simple,
    get_closest_line,
    get_further_outer_baseline_corner,
    group_lines,
    image_to_lines,
    is_court_corner,
    is_inner_sideline,
    transform_intersection,
    transform_line,
    transform_point,
    traverse_line,
)
from .lines import Intersection, Line, Point


class CourtFinder:
    """
    This class provides functionality for detecting and locating project-relevant objects on a tennis court.

    This class is designed to locate and store intersections between court lines,
    and map them to their respective lines for further geometric or object-detection
    processing. Intersections are sorted by their vertical (Y-axis) coordinate in
    descending order, so that higher points on the image are processed first.

    Attributes:
        img : np.ndarray
            The image (as a NumPy array) of the tennis court on which detection is performed.
        offset : int
            A pixel offset used in downstream calculations (e.g., for tolerance in matching
            or visualization).
        intersections : list[Intersection]
            A list of Intersection objects sorted from top to bottom (highest Y to lowest Y).
        line_intersection_mapping : dict[Line, set[Intersection]]
            A mapping from each court line to the set of intersections that lie on it.
    """

    def __init__(
        self, intersections: list[Intersection], img: np.ndarray, corner_offset: int = 30, netline_offset: int = 20
    ) -> None:
        """
        Initialize the CourtFinder with intersection data, image, and offset.

        Parameters:
            intersections : list[Intersection]
                A list of intersection objects, each containing two lines and a point of intersection.
                These will be sorted by their Y coordinate in descending order (top to bottom in image space).
            img : np.ndarray
                The court image to process.
            offset : int, optional
                Pixel offset for tolerance-based calculations (default is 20).

        Notes:
            During initialization:
            - Intersections are sorted by their Y coordinate in descending order.
            - A mapping (`line_intersection_mapping`) is built to quickly find all intersections
            belonging to a given court line.
        """
        self.img = img
        self.corner_offset = corner_offset
        self.netline_offset = netline_offset
        self.intersections = sorted(set(intersections), key=lambda intersection: -intersection.point.y)
        self.line_intersection_mapping: dict[Line, set[Intersection]] = defaultdict(set)

        for intersect in self.intersections:
            for line in (intersect.line1, intersect.line2):
                self.line_intersection_mapping[line].add(intersect)

    def find_closer_outer_baseline_point(self) -> tuple[Intersection, Intersection, Line]:
        """
        Finds a baseline-side intersection and its nearest corresponding net-side point.

        This method iterates through all intersections and filters those whose angle 
        lies between 90 and 270 degrees (i.e., outer baseline side). For each qualifying intersection, 
        it explores both connected lines to locate the nearest intersection along that line 
        that also meets the angular condition.

        Once a candidate line is found, it verifies whether the intersection belongs to 
        a valid court corner region using `find_point_neighbourhood` and `is_court_corner`. 
        If so, it proceeds to search for the corresponding intersection point 
        closer to the net by calling `_find_closer_outer_netpoint`.

        Returns:
            tuple[Intersection, Point, Line]:
                - The qualifying outer-baseline intersection.
                - The corresponding net-side intersection point found along the same line.
                - The line used to reach that net-side point.

        Notes:
            - The `Intersection` objects must have attributes: `angle`, `line1`, `line2`, and `point`.
            - The method relies on `self.line_intersection_mapping` for retrieving 
            intersections associated with each line.
            - Returns `None` implicitly if no valid baseline–net intersection pair is found.
        """
        for intersect in self.intersections:
            if not 270 > intersect.angle > 90:
                continue

            nearest_intersection = None
            for line in (intersect.line1, intersect.line2):
                sorted_intersection_points = sorted(
                    self.line_intersection_mapping[line], key=lambda intersection: intersection.distance(intersect)
                )

                for inner_intersect in sorted_intersection_points:
                    if not 270 > inner_intersect.angle > 90:
                        continue
                    else:
                        nearest_intersection = inner_intersect
                        break

                if nearest_intersection is not None:
                    img_piece, *original_range = find_point_neighbourhood(
                        intersect.point, self.corner_offset, self.img, line
                    )

                    if not is_court_corner(img_piece, intersect.point, original_range):
                        continue

                    net_intersection, closer_outer_baseline_point_used_line = self._find_closer_outer_netpoint(
                        line, intersect.point
                    )

                    if net_intersection is not None:
                        return intersect, net_intersection, closer_outer_baseline_point_used_line

    def _find_closer_outer_netpoint(self, line: Line, point: Point, warmup: int = 5) -> tuple[Intersection, Line]:
        """Traverse from a starting point to the net and return the first net intersection.

        This helper advances from point along line (skipping an initial `warmup`)
        and evaluates net-line candidates in cropped stripes. When a valid net segment
        is found, it returns both the geometric intersection and the supporting net
        Line instance.

        Args:
            line: Direction of travel toward the net.
            point: Starting point near the closer side of the court.
            warmup: Number of initial steps to ignore to avoid local noise.

        Returns:
            tuple[Intersection, Line] | None: The intersection between the scan and the
            detected net line (plus the net `Line` itself), or None if no net is detected.
        """
        net_intersection = None
        intersection_global = None
        i = 0
        while net_intersection is None:
            i += 1
            new_point, img_piece, original_range = traverse_line(point, self.netline_offset, self.img, line)

            if new_point.y >= point.y:
                break
            else:
                point = new_point

            if i < warmup:
                continue

            if is_inner_sideline(img_piece):
                break

            net_line_groups = find_net_lines(img_piece)

            intersections = []
            local_line = transform_line(line, self.img, *original_range, False)

            SETTINGS.debug and _plot_objs(img_piece)

            if not check_items_sign(net_line_groups):
                for net_line in net_line_groups:
                    intersection = local_line.intersection(net_line, img_piece)

                    if SETTINGS.debug:
                        draw_and_display(img_piece, lines=[local_line, net_line])

                    if intersection is not None and np.sign(net_line.slope) != np.sign(local_line.slope):
                        intersections.append(intersection)

            if len(intersections) > 0:
                net_intersection = sorted(intersections, key=lambda intersection: intersection.point.y)[-1]
                intersection_global = transform_intersection(net_intersection, self.img, *original_range)

        return intersection_global, line

    def find_further_outer_baseline_intersection(
        self,
        closer_outer_baseline_intersection: Intersection,
        used_line: Line,
        cannys_lower_thresh: int = 20,
        cannys_upper_thresh: int = 100,
        warmup: int = 5,
        offset: int = None,
    ) -> Intersection | None:
        """Find the further outer baseline by continuing the scan past the court.

        Starting from the known closer-baseline intersection, this method keeps
        moving along `used_line` and searches for the opposite (further) outer
        baseline using Canny + Hough with configurable thresholds.

        Args:
            closer_outer_baseline_intersection: Already detected closer-side
                baseline intersection.
            used_line: The scan line along which to continue searching.
            cannys_lower_thresh: Lower Canny threshold.
            cannys_upper_thresh: Upper Canny threshold.
            warmup: Number of initial iterations to ignore before evaluating.
            offset: Pixel step; if None, a sensible default based on image size is used.

        Returns:
            Intersection | None: Intersection with the further outer baseline, or None if not found.
        """
        further_outer_baseline_intersection = None
        offset = self.netline_offset if not offset else offset
        point = closer_outer_baseline_intersection.point
        unused_line = closer_outer_baseline_intersection.other_line(used_line)
        i = 0
        while True:
            i += 1
            new_point, img_piece, original_range = traverse_line(point, offset, self.img, unused_line)
            local_line = transform_line(unused_line, self.img, *original_range, False)

            if new_point.y >= point.y:
                break
            else:
                point = new_point

            if i < warmup:
                continue

            further_outer_baseline_intersection = get_further_outer_baseline_corner(
                img_piece, local_line, cannys_lower_thresh, cannys_upper_thresh
            )

            if further_outer_baseline_intersection:
                break
            else:
                continue

        return transform_intersection(further_outer_baseline_intersection, self.img, *original_range), local_line

    def find_netline(
        self,
        closer_outer_netpoint: Point,
        baseline: Line,
        max_line_gap: int,
        cannys_thresh_lower: int = 50,
        cannys_thresh_upper: int = 150,
        hough_thresh: int = 100,
        min_line_len: int = 100,
    ) -> Line:
        """Detect the net line.

        The search is constrained around `closer_outer_netpoint` and the
        `baseline`. Edges are extracted with Canny, and straight segments are
        grouped via probabilistic Hough; the best candidate aligned with the
        expected net orientation is returned.

        Args:
            closer_outer_netpoint: A point on/near the net on the closer side.
            baseline: Reference baseline used to bound the search region.
            max_line_gap: Maximum gap between line segments to link (Hough).
            cannys_thresh_lower: Lower Canny threshold.
            cannys_thresh_upper: Upper Canny threshold.
            hough_thresh: Minimum votes required in Hough space.
            min_line_len: Minimum line length to accept (in pixels).

        Returns:
            Line: The detected net line as a Line instance.
        """
        pic_copy = self.img.copy()
        img_gray = cv2.cvtColor(pic_copy, cv2.COLOR_RGB2GRAY)
        inv_img_gray = 255 - img_gray
        edges = cv2.Canny(inv_img_gray, cannys_thresh_lower, cannys_thresh_upper)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap
        )
        lines = [] if lines is None else lines

        line_obj = [Line.from_hough_line(line[0]) for line in lines]
        line_obj = [
            line for line in line_obj if line.slope is not None and np.sign(line.slope) == np.sign(baseline.slope)
        ]

        return get_closest_line(line_obj, closer_outer_netpoint)

    def find_further_doubles_sideline(
        self,
        further_outer_baseline_point: Point,
        prev_local_line: Line,
        offset: int,
        extra_offset: int,
        bin_thresh: float,
        surface_type: str,
    ) -> Line:
        """Detect the further doubles sideline (far side of the court).

        The method initializes a local scan near `further_outer_baseline_point`,
        biases the search using `prev_local_line` slope/angle, applies binary
        thresholding (`bin_thresh`) depending on `surface_type` (e.g., clay,
        another), and consolidates Hough segments into a single sideline.

        Args:
            further_outer_baseline_point: Point on the far outer baseline used as a seed.
            prev_local_line: Previously detected local line to guide expected orientation.
            offset: Primary sampling step along the scan direction.
            extra_offset: Occasional larger step to escape weak responses.
            bin_thresh: Threshold used for binarization before edge detection.
            surface_type: String/enum describing the court surface to tune thresholds.

        Returns:
            Line: The detected further doubles sideline
        """
        img_piece, *original_range = find_point_neighbourhood_simple(
            further_outer_baseline_point, offset + extra_offset, self.img, prev_local_line
        )

        SETTINGS.debug and _plot_objs(img_piece)

        img_gray = cv2.cvtColor(img_piece, cv2.COLOR_RGB2GRAY)

        if surface_type == "clay":

            line_obj = detect_lines_opposite_slope(
                img_gray,
                150, 500,
                hough_threshold=20,
                min_line_length=50,
                max_line_gap=10,
                ref_line=prev_local_line,
                )

            local_point = transform_point(further_outer_baseline_point, *original_range, False)
            line = get_closest_line(line_obj, local_point)

            if SETTINGS.debug:
                draw_and_display(img_piece, lines=[line])

            return transform_line(line, self.img, *original_range)

        bin_img = (img_gray > img_gray.max() * bin_thresh).astype(np.uint8)
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [
            con for con in contours if len(con) > 0 and (con[:, 0, 1].min() != 0 or con[:, 0, 0].min() != 0)
        ]

        if filtered_contours:
            areas = [cv2.contourArea(con) for con in filtered_contours]
            biggest_contour = filtered_contours[areas.index(max(areas))]

        img_copy = img_piece.copy()
        detected_obj = cv2.drawContours(np.zeros_like(img_copy), contours, areas.index(max(areas)), 255, thickness=1)

        mask = np.zeros_like(bin_img, dtype=np.uint8)
        cv2.fillPoly(mask, [biggest_contour], 1)

        SETTINGS.debug and _plot_objs(bin_img, detected_obj, mask)

        ones = np.argwhere(mask == 1)
        if prev_local_line.slope < 0:
            x_max = ones[:, 1].max()
            rightmost_indices = ones[:, 1] == x_max
            leftmost_indices = ones[:, 1] == 0
            y1, x1 = ones[rightmost_indices][-1]
            y2, x2 = ones[leftmost_indices][0]

        else:
            x_min = ones[:, 1].min()
            leftmost_indices = ones[:, 1] == x_min
            rightmost_indices = ones[:, 1] == mask.shape[1] - 1
            y1, x1 = ones[leftmost_indices][0]
            y2, x2 = ones[rightmost_indices][0]

        p1, p2 = (x1, y1), (x2, y2)

        if SETTINGS.debug:
            draw_and_display(img_piece, points=[p1, p2])

        line = Line.from_points(p1, p2)
        return transform_line(line, self.img, *original_range)

    def scan_endline(
        self,
        baseline: Line,
        netline: Line,
        closer_outer_baseline_point: Point,
        further_outer_netpoint: Point,
        closer_outer_netpoint: Point,
        further_outer_baseline_point: Point,
        bin_thresh: float,
        cannys_lower_thresh: int,
        cannys_lower_upper: int,
        hough_max_line_gap: int,
        hough_thresh: int = 20,
        warmup: int = 2,
        further_outer_endline_point_tolerance: int = 5,
        searching_line: Literal["net", "base"] = "base",
    ) -> tuple[Point, Point]:
        """
        Scans along the selected endline (baseline or netline) to detect the inner endline pair and
        returns their global intersections with the endline.

        The method iteratively traverses image windows along `endline` (after an optional `warmup`),
        binarizes each window, detects lines with slope opposite to `endline` (Canny → Hough), groups
        them, and keeps candidates whose intersections with `netline` lie between the outer net points
        and whose intersections with `endline` are not within a tolerance of the given further-outer
        endline point. For each window it records the group's `min` and `max` lines; after scanning to
        the far end, the first `min` and last `max` intersections define the closer/further inner
        endline points.

        Args:
            baseline, netline (Line): Court reference lines.
            closer_outer_baseline_point, further_outer_baseline_point (Point): Outer baseline anchors.
            closer_outer_netpoint, further_outer_netpoint (Point): Outer netline anchors.
            bin_thresh (float): Fraction of local max grayscale used for binarization.
            cannys_lower_thresh (int): Canny lower threshold.
            cannys_lower_upper (int): Canny upper threshold.
            hough_max_line_gap (int): Max gap for probabilistic Hough.
            hough_thresh (int): Hough accumulator threshold.
            warmup (int): Number of initial traverse steps to skip before detecting.
            further_outer_endline_point_tolerance (int): Pixel tolerance around `further_outer_point`
                used to exclude near-duplicate intersections.
            searching_line (Literal["net","base"]): Which endline to scan; selects (new_point, endline,
                further_outer_point) from netline or baseline context.

        Returns:
            tuple[Point, Point]: (closer_inner_endline_point, further_inner_endline_point).

        Notes:
            Assumes at least one `min` and one `max` candidate are found; otherwise the method will fail.
        """
        options = {
            "net": {
                "new_point": closer_outer_netpoint,
                "endline": netline,
                "further_outer_pt": further_outer_netpoint,
            },
            "base": {
                "new_point": closer_outer_baseline_point,
                "endline": baseline,
                "further_outer_pt": further_outer_baseline_point,
            },
        }[searching_line]

        new_point = options["new_point"]
        endline = options["endline"]
        further_outer_point = options["further_outer_pt"]
        endline_points_max = []
        endline_points_min = []

        i = 0
        while True:
            i += 1
            new_point, img_piece, (origin_x, origin_y) = traverse_line(
                new_point, self.corner_offset, self.img, endline, neighbourhood_type="simple"
            )
            pt_end = Point(origin_x + img_piece.shape[1], origin_y + img_piece.shape[0])

            if i <= warmup:
                continue

            img_gray = cv2.cvtColor(img_piece, cv2.COLOR_RGB2GRAY)
            local_endline = transform_line(endline, self.img, origin_x, origin_y, False)
            bin_img = (img_gray > img_gray.max() * bin_thresh).astype(np.uint8) * 255

            if searching_line == "net":
                net_img = np.zeros(img_piece.shape[:2], dtype=np.uint8)
                local_pts = local_endline.limit_to_img(net_img)
                net_img = cv2.line(net_img, *local_pts, 255, 1)

                bin_img |= net_img

            line_obj = detect_lines_opposite_slope(
                bin_img,
                cannys_lower_thresh, cannys_lower_upper,
                hough_threshold=hough_thresh,
                min_line_length=self.corner_offset * 0.5,
                max_line_gap=hough_max_line_gap,
                ref_line=endline,
            )

            grouped_lines = group_lines(line_obj)

            SETTINGS.debug and _plot_objs(img_piece, bin_img)

            local_endline = transform_line(endline, self.img, origin_x, origin_y, False)

            for line in grouped_lines:
                for line_type in ("min", "max"):
                    local_line = line.get_line(line_type)
                    global_line = transform_line(local_line, self.img, origin_x, origin_y)
                    test_intersection = global_line.intersection(netline, self.img)

                    if test_intersection is not None:
                        test_intersection_point = test_intersection.point
                    else:
                        continue

                    if (
                        closer_outer_netpoint.x < test_intersection_point.x < further_outer_netpoint.x
                        or closer_outer_netpoint.x > test_intersection_point.x > further_outer_netpoint.x
                    ):
                        endline_intersection_point = global_line.intersection(endline, self.img).point

                        local_intersection = local_line.intersection(local_endline, img_piece)

                        if (local_intersection is not None) and not (
                            endline_intersection_point.x - further_outer_endline_point_tolerance
                            < further_outer_point.x
                            < endline_intersection_point.x + further_outer_endline_point_tolerance
                            and further_outer_point.y - further_outer_endline_point_tolerance
                            < endline_intersection_point.y
                            < further_outer_point.y + further_outer_endline_point_tolerance
                        ):
                            if line_type == "min":
                                endline_points_min.append((endline_intersection_point, test_intersection_point))
                            else:
                                endline_points_max.append((endline_intersection_point, test_intersection_point))

            img_copy = img_piece.copy()
            for line in grouped_lines:
                pts = line.limit_to_img(img_copy)
                pts_base = local_endline.limit_to_img(img_copy)
                cv2.line(img_copy, *pts, (255, 0, 0), 1)
                cv2.line(img_copy, *pts_base, (0, 255, 0), 1)

            SETTINGS.debug and _plot_objs(img_copy)

            if further_outer_point.is_in_area(Point(origin_x, origin_y), pt_end):
                break

        closer_inner_endline_point = endline_points_min[0][0]
        further_inner_endline_point = endline_points_max[-1][0]

        return closer_inner_endline_point, further_inner_endline_point

    def find_net_service_point_centre_service_line(
        self,
        closer_outer_baseline_point: Point,
        closer_outer_netline_point: Point,
        further_outer_baseline_point: Point,
        further_outer_netline_point: Point,
        closer_inner_baseline_point: Point,
        further_inner_baseline_point: Point,
        closer_inner_netline_point: Point,
        further_inner_netline_point: Point,
        baseline: Line,
        netline: Line,
        bin_thresh: float,
        cannys_lower_thresh: int,
        cannys_upper_thresh: int,
        hough_max_line_gap: int,
        min_line_len_ratio: float,
        hough_thresh: int = 100,
        margin: int = 10,
        same_slope_sign: bool = False,
    ) -> tuple[Point, Line]:
        """
        Detects the service-line candidate that crosses both the netline and baseline, and returns
        its intersection on the netline.

        The method crops the court region between given baselines/netlines, detects candidate lines
        (binarization → Canny → Hough), filters out known court lines, and selects a line whose
        intersections with the netline and baseline fall between the inner boundary points (with margin).
        Optionally constrains detection to lines sharing the baseline’s slope sign.

        Args:
            closer_outer_baseline_point, closer_outer_netline_point, further_outer_baseline_point,
            further_outer_netline_point, closer_inner_baseline_point, further_inner_baseline_point,
            closer_inner_netline_point, further_inner_netline_point (Point): Key court boundary points.
            baseline, netline (Line): Reference court lines used for intersection tests.
            bin_thresh (float): Threshold for image binarization.
            cannys_lower_thresh, cannys_upper_thresh (int): Canny edge thresholds.
            hough_max_line_gap (int): Max segment gap for probabilistic Hough.
            min_line_len_ratio (float): Minimum line length ratio for detection.
            hough_thresh (int): Hough accumulator threshold.
            margin (int): Pixel margin for intersection windowing.
            same_slope_sign (bool): If True, keep only lines with the same slope sign as `baseline`.

        Returns:
            tuple[Point, Line]: (net_service_point_on_netline, detected_service_line) if found, else None.
        """
        img_piece, *origin = crop_court_field(
            self.img,
            baseline,
            closer_outer_baseline_point,
            closer_outer_netline_point,
            further_outer_baseline_point,
            further_outer_netline_point,
        )
        grouped_lines = image_to_lines(
            img_piece,
            bin_thresh,
            cannys_lower_thresh,
            cannys_upper_thresh,
            hough_thresh,
            self.corner_offset * min_line_len_ratio,
            hough_max_line_gap,
            baseline,
            same_slope_sign,
        )

        global_grouped_lines = [transform_line(line, self.img, *origin) for line in grouped_lines]
        global_grouped_lines = [
            line
            for line in global_grouped_lines
            if not (
                line.check_point_on_line(closer_outer_baseline_point, 30)
                or line.check_point_on_line(closer_outer_netline_point, 30)
                or line.check_point_on_line(further_outer_baseline_point, 30)
                or line.check_point_on_line(further_outer_netline_point, 30)
                or line.check_point_on_line(closer_inner_baseline_point, 30)
                or line.check_point_on_line(further_inner_baseline_point, 30)
                or line.check_point_on_line(closer_inner_netline_point, 30)
                or line.check_point_on_line(further_inner_netline_point, 30)
            )
        ]

        if SETTINGS.debug:
            draw_and_display(self.img, lines=global_grouped_lines)

        for line in global_grouped_lines:
            if (net_inter := line.intersection(netline, self.img)) is not None and (
                base_inter := line.intersection(baseline, self.img)
            ) is not None:
                slope_positive = baseline.slope > 0

                netline_condition = (
                    slope_positive
                    and further_inner_netline_point.x + margin
                    < net_inter.point.x
                    < closer_inner_netline_point.x - margin
                ) or (
                    not slope_positive
                    and closer_inner_netline_point.x + margin
                    < net_inter.point.x
                    < further_inner_netline_point.x - margin
                )

                baseline_condition = (
                    slope_positive
                    and further_inner_baseline_point.x + margin
                    < base_inter.point.x
                    < closer_inner_baseline_point.x - margin
                ) or (
                    not slope_positive
                    and closer_inner_baseline_point.x + margin
                    < base_inter.point.x
                    < further_inner_baseline_point.x - margin
                )

                if SETTINGS.debug:
                    draw_and_display(self.img, points=[net_inter.point, base_inter.point])

                if netline_condition and baseline_condition:
                    return net_inter.point, line

    def find_center(
        self,
        closer_outer_baseline_point: Point,
        closer_outer_netline_point: Point,
        further_outer_baseline_point: Point,
        further_outer_netline_point: Point,
        closer_inner_baseline_point: Point,
        further_inner_baseline_point: Point,
        closer_inner_netline_point: Point,
        further_inner_netline_point: Point,
        baseline: Line,
        closer_inner_sideline: Line,
        further_inner_sideline: Line,
        centre_service_line: Line,
        bin_thresh: float,
        cannys_lower_thresh: int,
        cannys_upper_thresh: int,
        hough_max_line_gap: int,
        min_line_len_ratio: float,
        hough_thresh: int = 100,
        margin: int = 10,
    ) -> tuple[Point, Point, Point] | None:
        """
        Detects and returns the center service line intersection points within a tennis court image.

        The method crops the court region between given baselines and netlines, detects candidate lines
        using edge detection and Hough transform, filters out known court lines, and finds the line 
        crossing both inner sidelines and the center service line.

        Args:
            closer_outer_baseline_point, closer_outer_netline_point, further_outer_baseline_point,
            further_outer_netline_point, closer_inner_baseline_point, further_inner_baseline_point,
            closer_inner_netline_point, further_inner_netline_point (Point): Key court boundary points.
            baseline, closer_inner_sideline, further_inner_sideline, centre_service_line (Line): 
                Reference court lines.
            bin_thresh (float): Threshold for image binarization.
            cannys_lower_thresh, cannys_upper_thresh (int): Canny edge detection thresholds.
            hough_max_line_gap (int): Max gap between segments in Hough transform.
            min_line_len_ratio (float): Minimum line length ratio for detection.
            hough_thresh (int): Hough transform threshold.
            margin (int): Pixel margin for filtering intersections.

        Returns:
            tuple[Point, Point, Point]: (center_service_point, further_service_point, closer_service_point)
                if found, otherwise None.
        """   
        img_piece, *origin = crop_court_field(
            self.img,
            baseline,
            closer_outer_baseline_point,
            closer_outer_netline_point,
            further_outer_baseline_point,
            further_outer_netline_point,
        )
        grouped_lines = image_to_lines(
            img_piece,
            bin_thresh,
            cannys_lower_thresh,
            cannys_upper_thresh,
            hough_thresh,
            self.corner_offset * min_line_len_ratio,
            hough_max_line_gap,
            closer_inner_sideline,
        )

        global_grouped_lines = [transform_line(line, self.img, *origin) for line in grouped_lines]
        global_grouped_lines = [
            line
            for line in global_grouped_lines
            if not (
                line.check_point_on_line(closer_outer_baseline_point, 30)
                or line.check_point_on_line(closer_outer_netline_point, 30)
                or line.check_point_on_line(further_outer_baseline_point, 30)
                or line.check_point_on_line(further_outer_netline_point, 30)
                or line.check_point_on_line(closer_inner_baseline_point, 30)
                or line.check_point_on_line(further_inner_baseline_point, 30)
                or line.check_point_on_line(closer_inner_netline_point, 30)
                or line.check_point_on_line(further_inner_netline_point, 30)
            )
        ]

        if SETTINGS.debug:
            draw_and_display(self.img, lines=global_grouped_lines)

        for line in global_grouped_lines:
            if (closer_inter := line.intersection(closer_inner_sideline, self.img)) is not None and (
                further_inter := line.intersection(further_inner_sideline, self.img)
            ) is not None:
                slope_positive = closer_inner_sideline.slope > 0

                closer_line_condition = (
                    slope_positive
                    and closer_inner_netline_point.x + margin
                    < closer_inter.point.x
                    < closer_inner_baseline_point.x - margin
                ) or (
                    not slope_positive
                    and closer_inner_baseline_point.x + margin
                    < closer_inter.point.x
                    < closer_inner_netline_point.x - margin
                )

                further_line_condition = (
                    slope_positive
                    and further_inner_netline_point.x + margin
                    < further_inter.point.x
                    < further_inner_baseline_point.x - margin
                ) or (
                    not slope_positive
                    and further_inner_baseline_point.x + margin
                    < further_inter.point.x
                    < further_inner_netline_point.x - margin
                )

                if closer_line_condition and further_line_condition:
                    if SETTINGS.debug:
                        draw_and_display(self.img, lines=[line])

                    centre_service_point = line.intersection(centre_service_line, self.img).point
                    further_service_point = line.intersection(further_inner_sideline, self.img).point
                    closer_service_point = line.intersection(closer_inner_sideline, self.img).point

                    return centre_service_point, further_service_point, closer_service_point
                

def process[T: BaseModel](pic: np.ndarray, param: T) -> tuple[dict[str, Point] | None, dict[str, Line] | None, dict[str, dict[str, float]] | None]:
    """
    Detects and reconstructs tennis court geometry from an input image.

    Uses Hough transform to extract court lines, groups and intersects them,
    and employs CourtFinder logic to locate baseline, netline, sidelines,
    and service lines based on parameter settings.

    Workflow:
    For each image:
        - Detect lines (Canny + Hough), build `Line` objects, filter verticals (`xv is None`), and group lines.
        - Compute all group–group intersections and initialize `CourtFinder`.
        - Find outer baseline/net points and sidelines; derive baseline and net line.
        - Scan for inner (singles) baselines and netline endpoints.
        - Derive centre service line, net service point, court centre, and service line.
        - Compose destination points/lines and plot/save results.

    Args:
        pic: Input court image as a NumPy array.
        param: Pydantic model instance with parameters to proceed the image

    Returns:
        A tuple containing:
            - dict of named Points (court keypoints) or None
            - dict of named Lines (court lines) or None
            - optional dict of ground-truth points if provided, else None
    """
    _, line_endpoints = apply_hough_transformation(pic)
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

    court_finder = CourtFinder(intersections, pic)

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
            param.canny_thresh.lower,
            param.canny_thresh.upper,
            offset=param.offset,
        )
        further_outer_baseline_point = further_outer_baseline_intersection.point

        baseline = Line.from_points(closer_outer_baseline_point, further_outer_baseline_point)

        netline = court_finder.find_netline(closer_outer_netline_point, baseline, param.max_line_gap)

        further_outer_sideline = court_finder.find_further_doubles_sideline(
            further_outer_baseline_point,
            last_local_line,
            param.offset,
            param.extra_offset,
            param.bin_thresh,
            param.surface_type,
        )
        further_outer_netline_point = further_outer_sideline.intersection(netline, pic).point

        closer_inner_baseline_point, further_inner_baseline_point = court_finder.scan_endline(
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

        closer_inner_netline_point, further_inner_netline_point = court_finder.scan_endline(
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
            param.bin_thresh_centre_service_line,
            param.canny_thresh.lower,
            param.canny_thresh.upper,
            param.max_line_gap_centre_service_line,
            param.min_line_len_ratio,
            param.hough_thresh,
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
            param.bin_thresh_centre_service_line,
            param.canny_thresh.lower,
            param.canny_thresh.upper,
            param.max_line_gap_centre_service_line,
            param.min_line_len_ratio,
            param.hough_thresh,
        )

        service_line = Line.from_points(further_service_point, closer_service_point)

        common_args = (
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

        has_gt = getattr(param, "ground_truth_points", None) is not None

        if has_gt:
            dst_points, dst_lines, ground_truth_points = compose_court_data(*common_args, param)
            return dst_points, dst_lines, ground_truth_points
        
        dst_points, dst_lines = compose_court_data(*common_args)
        return dst_points, dst_lines, None

    except Exception as e:
        print(e)
        return None, None, None
