import colorsys
import json
import os
import random
from pathlib import Path
from typing import Iterable, Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, TypeAdapter, ValidationError
from skimage.morphology import skeletonize

from .const import (
    ARRAY_X_INDEX,
    ARRAY_Y_INDEX,
    COURT_LENGTH_HALF,
    COURT_WIDTH_HALF,
    DIST_FROM_BASELINE,
    DIST_OUTER_SIDELINE,
    LENGTH,
    SETTINGS,
    WIDTH,
)
from .lines import Intersection, Line, LineGroup, Point
from .schemas import GroundTruthCourtPoints


def _plot_objs(*objs: np.ndarray) -> None:
    """
    Displays one or more image-like NumPy arrays sequentially using Matplotlib.

    Each provided array is shown in a separate figure window for quick
    visualization, typically used for debugging intermediate image
    processing results.

    Args:
        *objs (np.ndarray): One or more image arrays to display.
    """
    for obj in objs:
        plt.imshow(obj)
        plt.show()


def get_pictures(path: str) -> dict[str, list[np.ndarray]]:
    """
    Loads all valid image files from a directory and returns their representations in RGB, HSV, and grayscale formats.
    Only files with extensions '.jpg', '.jpeg', '.png' are considered. Unreadable files are skipped.

    Args:
        path (str): path to pictures directory

    Returns:
        dict: {
            'rgb': list of images in RGB format,
            'hsv': list of images in HSV format,
            'gray': list of images in grayscale format
        }
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    filenames = sorted(
        [
            f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in valid_extensions
        ]
    )

    pics = {"rgb": [], "hsv": [], "gray": []}

    for filename in filenames:
        img_path = os.path.join(path, filename)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            continue

        pics["rgb"].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pics["hsv"].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
        pics["gray"].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    return pics


def apply_hough_transformation(
    img_rgb: np.ndarray,
    blur_kernel_size: int = 5,
    canny_thresh_lower: int = 50,
    canny_thresh_upper: int = 150,
    hough_thresh: int = 100,
    hough_min_line_len: int = 100,
    hough_max_line_gap: int = 10,
) -> tuple[np.ndarray, list]:
    """
    It applies probabilistic Hough line transformation to given RGB image
    process of applying hough transformation to the image is as follows:
        - creating copy of original image,
        - then copy is converted to gray scale,
        - later the gaussian blur filter is applied to the gray scaled image
        - on the blurred image Canny's edges detection is processed
        - on the binary image that contains the detected edges Hough transformation is processed to get lines,
            if any detected, it contains two points - both ends of line, between them we can plot strainght line

    In HoughLinesP function - constant values of 1 and np.pi/180 indicates
        respectively rho and theta parameters that dont need to be tuned

    Args:
        img_rgb (np.ndarray): an RGB image
        blur_kernel_size (int, optional): size of square blur kernel. Defaults to 5.
        canny_thresh_lower (int, optional): lower threshold for the hysteresis process in Canny. Defaults to 50.
        canny_thresh_upper (int, optional): upper thresholds for the hysteresis process in Canny. Defaults to 150.
        hough_thresh (int, optional): minimum number of intersections (votes) in
            the accumulator to "declare" a line - the higher value the fewer lines, only strong ones. Defaults to 100.
        hough_min_line_len (int, optional): the minimum length (in pixels) of a
            line segment to be accepted, short segments below this length are ignored. Defaults to 100.
        hough_max_line_gap (int, optional): the maximum allowed gap between two
            line segments to treat them as a single line, if endpoints of two
            segments are close enough (within this gap), they are joined into one line. Defaults to 10.

    Returns:
        tuple[np.ndarray, list]: image with detected lines drawn,
        list of lists of 4 integers each item - that indicates both ends of detected lines
    """
    img_copy = img_rgb.copy()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_thresh_lower, canny_thresh_upper)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=hough_thresh, minLineLength=hough_min_line_len, maxLineGap=hough_max_line_gap
    )
    if lines is None:
        lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_copy, lines


def group_lines(
    lines: list[Line], thresh_theta: float | int = 5, thresh_intercept: float | int = 10
) -> list[LineGroup]:
    """
    Group similar Line objects into LineGroups based on orientation and position thresholds.

    Args:
        lines (list[Line]): A list of Line objects to group.
        thresh_theta (float): Maximum allowed angle difference between lines to be in the same group.
        thresh_intercept (float): Maximum allowed intercept difference (for non-vertical lines).

    Returns:
        list[LineGroup]: A list of LineGroup objects representing grouped lines.
    """
    groups = []

    for line in lines:
        for group in groups:
            if group.process_line(line, thresh_theta, thresh_intercept):
                break
        else:
            groups.append(LineGroup([line]))

    return groups


def draw_line_group(
    img: np.ndarray,
    line_group: LineGroup,
    color: tuple[int, int, int],
    approx_color: tuple[int, int, int],
    approx_only: bool = True,
) -> np.ndarray:
    """
    Draw a LineGroup on an image.

    Args:
        img (np.ndarray): Input image to draw on.
        line_group (LineGroup): The LineGroup to visualize.
        color (tuple[int, int, int]): Color for individual lines (BGR).
        approx_color (tuple[int, int, int]): Color for the approximated line (BGR).
        approx_only (bool): If True, only draw the approximated line. Defaults to True.

    Returns:
        np.ndarray: A copy of the image with drawn lines.
    """
    img_copy = img.copy()

    if not approx_only:
        for line in line_group.lines:
            p1, p2 = line.limit_to_img(img_copy)
            cv2.line(img_copy, p1, p2, color, thickness=1)

    p1, p2 = line_group.limit_to_img(img_copy)
    cv2.line(img_copy, p1, p2, approx_color, thickness=2)
    return img_copy


def generate_similar_color_pairs(n: int = 10) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """
    Generate 'n' pairs of similar colors for visualization.
    Each pair shares the same hue and brightness, but differs in saturation.

    Args:
        n (int): Number of color pairs to generate. Defaults to 10.

    Returns:
        list: A list of (color1, color2) pairs in BGR format (integers in 0-255).
    """
    color_pairs = []
    for _ in range(n):
        h = random.random()
        v = random.uniform(0.6, 1.0)
        s_low = random.uniform(0.1, 0.4)
        s_high = random.uniform(0.7, 1.0)

        rgb1 = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s_low, v))
        rgb2 = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s_high, v))

        bgr1 = rgb1[::-1]
        bgr2 = rgb2[::-1]

        color_pairs.append((bgr1, bgr2))

    return color_pairs


def find_point_neighbourhood(point: Point, offset: int, img: np.ndarray, line: Line) -> tuple[np.ndarray, int, int]:
    """
    Extract a rectangular neighborhood around a given point, constrained by a line and image boundaries.
    If the height from the line's range is smaller than `offset`, it is expanded to meet the minimum height.
    The height based on the line is always used as priority when it is greater than or equal to `offset`.

    Args:
        point (Point):
            The central point from which the neighborhood is calculated.
        offset (int):
            Horizontal distance in pixels from the point in the x-direction,
            and the minimum allowed height of the neighborhood.
        img (np.ndarray):
            The source image array.
        line (Line):
            A line used to determine the vertical range from the horizontal limits.

    Returns:
        tuple[np.ndarray, int, int]:
            A tuple containing:
                - The extracted sub-image as a numpy array.
                - The x-coordinate of the top-left corner of the extracted region, value x of global origin.
                - The y-coordinate of the top-left corner of the extracted region, value y of global origin.
    """
    height, width = img.shape[ARRAY_Y_INDEX], img.shape[ARRAY_X_INDEX]

    def clamp_window_around(center: int, win_size: int, limit: int) -> tuple[int, int]:
        """Return (start, end) for a window of `win_size` centered at `center`, clipped to [0, limit-1]."""
        win_size = min(win_size, limit)
        half = win_size // 2
        start = int(center) - half
        end = start + win_size - 1

        if start < 0:
            end -= start
            start = 0
        if end >= limit:
            start -= end - (limit - 1)
            end = limit - 1
            if start < 0:
                start = 0
        return start, end

    if abs(line.slope) > 1:
        y_start = max(int(point.y) - offset, 0)
        y_end = min(int(point.y) + offset, height - 1)

        if y_start > y_end:
            y_start, y_end = y_end, y_start

        if line.slope is None or line.intercept is None:
            x_start = max(int(point.x) - offset, 0)
            x_end = min(int(point.x) + offset, width - 1)
        else:
            x0 = max(min(line.x_for_y(y_start), width - 1), 0)
            x1 = max(min(line.x_for_y(y_end), width - 1), 0)
            x_start, x_end = min(x0, x1), max(x0, x1)

            curr_w = x_end - x_start + 1
            if curr_w < offset:
                x_start, x_end = clamp_window_around(int(point.x), offset, width)

        if x_start > x_end:
            x_start, x_end = x_end, x_start

    else:
        x_start = max(int(point.x) - offset, 0)
        x_end = min(int(point.x) + offset, width - 1)

        if x_start > x_end:
            x_start, x_end = x_end, x_start

        if line.slope is None or line.intercept is None:
            y_start = max(int(point.y) - offset, 0)
            y_end = min(int(point.y) + offset, height - 1)
        else:
            y0 = max(min(line.y_for_x(x_start), height - 1), 0)
            y1 = max(min(line.y_for_x(x_end), height - 1), 0)
            y_start, y_end = min(y0, y1), max(y0, y1)

            curr_h = y_end - y_start + 1
            if curr_h < offset:
                y_start, y_end = clamp_window_around(int(point.y), offset, height)

        if y_start > y_end:
            y_start, y_end = y_end, y_start

    return img[y_start : y_end + 1, x_start : x_end + 1], x_start, y_start


def _clamp_to_img(p: Point, img: np.ndarray, line: Line) -> Point:
    """
    Clamp a point's coordinates so that it lies within the image boundaries.

    If the point is outside the image in either the x or y direction, it is
    projected back onto the image border along the given line. This ensures
    that all returned points are valid pixel positions.

    Parameters:
        p : Point
            The point to be clamped.
        img : np.ndarray
            The image array used to determine boundaries.
        line : Line
            The line object, used to compute the corresponding coordinate
            (x for a given y or y for a given x) when projecting onto the border.

    Returns:
        Point
            A new point lying within the image bounds.
    """
    if p.y >= img.shape[ARRAY_Y_INDEX]:
        y = img.shape[ARRAY_Y_INDEX] - 1
        return Point(line.x_for_y(y), y)
    if p.y < 0:
        y = 0
        return Point(line.x_for_y(y), y)
    if p.x < 0:
        x = 0
        return Point(x, line.y_for_x(x))
    if p.x >= img.shape[ARRAY_X_INDEX]:
        x = img.shape[ARRAY_X_INDEX] - 1
        return Point(x, line.y_for_x(x))
    return p


def _unit_tangent(line: Line) -> tuple[float, float]:
    """
    Compute the unit tangent vector of a line.

    The tangent vector is derived from the line slope (`line.slope`) or,
    in the case of a vertical line, is set to point straight up. The vector
    is normalized to have length 1.

    Parameters:
        line : Line
            The line object for which the tangent vector will be computed.
            Must provide a `slope` attribute.

    Returns:
        (float, float)
            The normalized (dx, dy) tangent vector.
    """
    m = getattr(line, "slope", None)
    if m is None or np.isinf(m):
        dx, dy = 0.0, 1.0
    else:
        dx, dy = 1.0, m
    norm = (dx * dx + dy * dy) ** 0.5
    if norm == 0:
        return (0.0, -1.0)
    return (dx / norm, dy / norm)


def traverse_line(
    point: Point,
    offset: int,
    img: np.ndarray,
    line: Line,
    direction: Literal["up", "down"] = "up",
    neighbourhood_type: Literal["complex", "simple"] = "complex",
) -> tuple[Point, np.ndarray, list[int, int]]:
    """
    Traverse along a given line from a starting point by a fixed offset,
    returning the next point in the specified direction, the extracted
    image neighborhood, and the global origin coordinates.

    This function:
        1. Finds candidate points along the line at the given offset from `point`.
        2. Clamps each candidate to the image boundaries (to avoid coordinates outside the image).
        3. Determines the unit tangent vector of the line.
        4. Adjusts the tangent's orientation based on the desired traversal direction
        ("up" for decreasing y on the image, "down" for increasing y).
        5. Selects the candidate point that best matches the desired movement direction
        using the dot product with the tangent vector.

    Parameters:
        point : Point
            The current point on the line from which traversal starts.
        offset : int
            The distance in pixels along the line to search for candidate points.
        img : np.ndarray
            The image array. Used to clamp points to valid pixel coordinates.
        line : Line
            The line object along which traversal occurs. Must provide slope,
            `x_for_y()` and `y_for_x()` methods, and `get_points_by_distance()`.
        direction : {"up", "down"}, default="up"
            The desired traversal direction:
            - "up": move toward smaller y-values (visually up in the image)
            - "down": move toward larger y-values (visually down in the image)

    Returns:
        new_point : Point
            The chosen point in the desired direction, clamped to image boundaries.
        img_piece : np.ndarray
            The extracted neighborhood of the original image around the starting point.
        global_origin : list[int, int]
            The [y, x] global origin coordinates of the extracted image piece.

    Notes:
        - This function automatically determines whether to prioritize x or y
        when choosing the next point, based on the line slope and direction.
        - No manual axis or index selection is required.
    """

    neighbourhood_func = {"simple": find_point_neighbourhood_simple, "complex": find_point_neighbourhood}[
        neighbourhood_type
    ]

    img_piece, *global_origin = neighbourhood_func(point, offset, img, line)

    points_candidates = line.get_points_by_distance(point, offset)
    points = [_clamp_to_img(p, img, line) for p in points_candidates]

    tx, ty = _unit_tangent(line)

    if direction == "up" and ty > 0:
        tx, ty = -tx, -ty
    elif direction == "down" and ty < 0:
        tx, ty = -tx, -ty

    def score(p: Point) -> float:
        vx, vy = (p.x - point.x), (p.y - point.y)
        return vx * tx + vy * ty

    new_point = max(points, key=score)
    return new_point, img_piece, global_origin


def find_net_lines(
    img_piece: np.ndarray,
    cannys_thresh_lower: int = 50,
    cannys_thresh_upper: int = 150,
    hough_thresh: int = 10,
    min_line_len: int = 10,
    max_line_gap: int = 10,
) -> list[LineGroup]:
    """
    Detects and groups net lines in a given image section.

    The function inverts the grayscale image, applies Canny edge detection,
    detects line segments using the probabilistic Hough transform, visualizes
    the detected lines (if debug mode is on), and groups them by slope and
    proximity.

    Args:
        img_piece (np.ndarray): RGB input image section containing the court net area.
        cannys_thresh_lower (int, optional): Lower Canny threshold (default: 50).
        cannys_thresh_upper (int, optional): Upper Canny threshold (default: 150).
        hough_thresh (int, optional): Threshold for Hough line detection (default: 10).
        min_line_len (int, optional): Minimum line segment length (default: 10).
        max_line_gap (int, optional): Maximum gap between line segments (default: 10).

    Returns:
        list[LineGroup]: List of grouped line objects representing detected net lines.
    """
    piece_gray = cv2.cvtColor(img_piece, cv2.COLOR_RGB2GRAY)
    neg_gray_img = 255 - piece_gray
    edges = cv2.Canny(neg_gray_img, cannys_thresh_lower, cannys_thresh_upper)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap
    )
    if lines is None:
        lines = []

    img_copy = img_piece.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    SETTINGS.debug and _plot_objs(piece_gray, neg_gray_img, edges, img_copy)

    line_obj = [Line.from_hough_line(line[0]) for line in lines]
    line_obj = [line for line in line_obj if line.slope is not None]
    return group_lines(line_obj)


def check_items_sign(line_groups: list[LineGroup]) -> bool:
    """
    Checks whether all lines in a list have slopes with the same sign.

    Returns True if all slopes are positive or all are negative, and False
    if the list contains a mix of positive and negative slopes.

    Args:
        line_groups (list[LineGroup]): List of line group objects with a `slope` attribute.

    Returns:
        bool: True if all slopes share the same sign, otherwise False.
    """
    return all(item.slope > 0 for item in line_groups) or all(item.slope < 0 for item in line_groups)


def transform_point(
    point: Intersection | Point, original_x_start: int, original_y_start: int, to_global: bool = True
) -> Point:
    """
    Transforms a point's coordinates between local and global image reference frames.

    The function shifts a point by the provided (x, y) offsets depending on the
    transformation direction. Works with both `Point` and `Intersection` objects.

    Args:
        point (Intersection | Point): Point or intersection to transform.
        original_x_start (int): X-axis offset.
        original_y_start (int): Y-axis offset.
        to_global (bool, optional): If True, converts from local to global coordinates;
                                    if False, converts from global to local (default: True).

    Returns:
        Point: Transformed point with updated coordinates.
    """
    if isinstance(point, Intersection):
        point = point.point

    if to_global:
        return Point(point.x + original_x_start, point.y + original_y_start)
    else:
        return Point(point.x - original_x_start, point.y - original_y_start)


def transform_line(
    original_line: Line, original_img: np.ndarray, original_x_start: int, original_y_start: int, to_global: bool = True
) -> Line:
    """
    Transforms a line's coordinates between local and global image reference frames.

    The function shifts both endpoints of a line by the provided offsets using
    `transform_point` and reconstructs a new line from the transformed coordinates.

    Args:
        original_line (Line): Line object to transform.
        original_img (np.ndarray): Image used to determine line limits.
        original_x_start (int): X-axis offset.
        original_y_start (int): Y-axis offset.
        to_global (bool, optional): If True, converts from local to global coordinates;
                                    if False, converts from global to local (default: True).

    Returns:
        Line: Transformed line object with updated coordinates.
    """
    pts_source: Iterable[Point] = original_line.limit_to_img(original_img)
    pts_transformed = [transform_point(p, original_x_start, original_y_start, to_global=to_global) for p in pts_source]
    return Line.from_points(*pts_transformed)


def transform_intersection(
    intersection: Intersection,
    source_img: np.ndarray,
    original_x_start: int,
    original_y_start: int,
    to_global: bool = True,
) -> Intersection:
    """
    Transforms an Intersection in one go.
    - If to_global=True: treats inputs as LOCAL and returns GLOBAL.
    - If to_global=False: treats inputs as GLOBAL and returns LOCAL.

    Note:
        `source_img` should be the image in the *source* space,
        i.e. the space you are transforming FROM. This keeps `limit_to_img`
        correct in both directions.
    """
    transformed_point = transform_point(intersection.point, original_x_start, original_y_start, to_global=to_global)

    line1_t = transform_line(intersection.line1, source_img, original_x_start, original_y_start, to_global)
    line2_t = transform_line(intersection.line2, source_img, original_x_start, original_y_start, to_global)
    return Intersection(line1_t, line2_t, transformed_point)


def _count_array_sequence_group(arr: np.ndarray) -> int:
    """
    Counts the number of consecutive value groups in a sorted integer array.

    A new group is started whenever the difference between adjacent elements
    exceeds 1. Commonly used to count separate contiguous sequences.

    Args:
        arr (np.ndarray): One-dimensional sorted array of integers.

    Returns:
        int: Number of contiguous value groups in the array.
    """
    counter = 0
    for i, item in enumerate(arr):
        if i > 0:
            if item - arr[i - 1] > 1:
                counter += 1
        else:
            counter += 1

    return counter


def is_court_corner(
    img: np.ndarray,
    intersect_point: Point,
    original_range: tuple[int, int],
    bin_thresh: float = 0.8,
    x_range: int = 3,
    y_range: int = 3,
) -> bool:
    """
    Determines whether a given intersection point corresponds to a court corner in an image.

    The function binarizes and morphologically closes the image, checks if the intersection
    point lies within a white region, and analyzes the shape of the connected white area
    to confirm a corner-like pattern.

    Args:
        img (np.ndarray): RGB input image.
        intersect_point (Point): Intersection point in real coordinates.
        original_range (tuple[int, int]): Image width and height for coordinate transformation.
        bin_thresh (float, optional): Binary threshold ratio (default: 0.8).
        x_range (int, optional): Horizontal margin for local check (default: 3).
        y_range (int, optional): Vertical margin for local check (default: 3).

    Returns:
        bool: True if the intersection matches a court corner pattern, otherwise False.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bin_img = (gray > gray.max() * bin_thresh).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed_bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    local_intersect_point = transform_point(intersect_point, *original_range, False)
    if (
        np.sum(
            closed_bin_img[
                local_intersect_point.y - y_range : local_intersect_point.y + y_range,
                local_intersect_point.x - x_range : local_intersect_point.x + x_range,
            ]
        )
        == 0
    ):
        return False

    img_copy = img.copy()
    cv2.circle(img_copy, local_intersect_point, 2, (0, 255, 0))

    SETTINGS.debug and _plot_objs(img, gray, bin_img, img_copy)

    ones_iloc = np.argwhere(closed_bin_img > 0)
    x_range = np.unique(ones_iloc[:, 1])
    y_range = np.unique(ones_iloc[:, 0])

    if len(x_range) == 0 or len(x_range) == closed_bin_img.shape[1] and len(y_range) == closed_bin_img.shape[0]:
        return False

    if not np.all(np.diff(x_range) == 1):
        return False

    row_start, row_stop = ones_iloc[:, 0].min(), ones_iloc[:, 0].max()
    seq_groups = []
    for row in range(row_start, row_stop + 1):
        ones = np.argwhere(closed_bin_img[row, :]).flatten()
        seq_num = _count_array_sequence_group(ones)

        if not seq_groups or seq_groups[-1] != seq_num:
            seq_groups.append(seq_num)

    if seq_groups != [1, 2, 1] and seq_groups != [2, 1]:
        return False

    return True


def angle_between_lines(line1: Line, line2: Line) -> float | None:
    """
    Calculates the smallest angle between two lines in degrees.

    The function handles vertical, horizontal, and sloped lines, returning None
    if both lines are vertical. It computes the absolute angular difference based
    on line slopes.

    Args:
        line1 (Line): First line object with a `slope` attribute.
        line2 (Line): Second line object with a `slope` attribute.

    Returns:
        float | None: Smallest angle between the two lines in degrees, or None if both are vertical.
    """
    if line1.slope is None and line2.slope is None:
        return None

    if line1.slope is None and line2.slope is not None:
        return np.degrees(np.arctan(abs(1 / line2.slope)))
    if line2.slope is None and line1.slope is not None:
        return np.degrees(np.arctan(abs(1 / line1.slope)))

    m1, m2 = line1.slope, line2.slope
    if 1 + m1 * m2 == 0:
        return 90.0

    tan_theta = abs((m1 - m2) / (1 + m1 * m2))
    return np.degrees(np.arctan(tan_theta))


def is_inner_sideline(
    img: np.ndarray,
    bin_thresh: float = 0.8,
    hough_line_thresh: int = 8,
    min_line_len: int | None = 5,
    min_line_gap: int = 5,
) -> bool:
    """
    Determines whether the given image represents an inner sideline of a court.

    The function binarizes and skeletonizes the image, detects lines using the
    probabilistic Hough transform, groups them by slope, and checks if at least
    two line groups with opposite slope directions form an acute angle (< 90°),
    indicating an inner sideline pattern.

    Args:
        img (np.ndarray): RGB input image.
        bin_thresh (float, optional): Binary threshold ratio (default: 0.8).
        hough_line_thresh (int, optional): Threshold for Hough line detection (default: 8).
        min_line_len (int | None, optional): Minimum line segment length (default: 5).
        min_line_gap (int, optional): Maximum allowed gap between line segments (default: 5).

    Returns:
        bool: True if the image corresponds to an inner sideline pattern, otherwise False.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bin_img = (gray > gray.max() * bin_thresh).astype(np.uint8)

    skel = skeletonize(bin_img).astype(np.uint8)

    lines = cv2.HoughLinesP(
        skel, 1, np.pi / 180, threshold=hough_line_thresh, minLineLength=min_line_len, maxLineGap=min_line_gap
    )
    if lines is None:
        lines = []

    img_copy = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    SETTINGS.debug and _plot_objs(img, gray, bin_img, skel, img_copy)

    line_obj = [Line.from_hough_line(line[0]) for line in lines]
    line_obj = [line for line in line_obj if line.slope is not None]
    line_groups = group_lines(line_obj)

    if len(set(np.sign(line.slope) for line in line_groups)) <= 1:
        return False

    angle = angle_between_lines(line_groups[0], line_groups[1])

    return angle is not None and angle < 90


def transform_annotation(img: np.ndarray, annotation: dict[Literal["x", "y"], float]) -> Point:
    """
    Converts a percentage-based annotation into pixel coordinates.

    The function transforms normalized annotation coordinates (in percent of
    image width and height) into absolute pixel coordinates within the given image.

    Args:
        img (np.ndarray): Input image used to determine width and height.
        annotation (dict[Literal["x", "y"], float]): Annotation with 'x' and 'y' values in percentages (0–100).

    Returns:
        Point: Transformed point in pixel coordinates.
    """
    height, width = img.shape[:2]
    x = annotation["x"] / 100 * width
    y = annotation["y"] / 100 * height
    return Point(x, y)


def fill_edges_image(edges_img: np.ndarray) -> np.ndarray:
    """
    Fills vertical gaps between detected edge pixels in a binary edge image.

    The function connects the topmost and bottommost edge pixels in each column,
    effectively filling enclosed regions, and applies morphological closing to
    smooth and finalize the filled area.

    Args:
        edges_img (np.ndarray): Binary edge image (values 0 or 1).

    Returns:
        np.ndarray: Binary image with filled vertical regions and closed gaps.
    """
    h, w = edges_img.shape
    filled = np.zeros_like(edges_img)

    for x in range(w):
        ys = np.flatnonzero(edges_img[:, x])
        if ys.size >= 2:
            y1, y2 = ys.min(), ys.max()
            filled[y1 : y2 + 1, x] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)


def _select_intersection_by_x(intersections: list[Intersection], local_line: Line) -> Intersection | None:
    """
    Selects an intersection point based on the x-coordinate and line slope.

    If the line has a positive slope, the intersection with the smallest x is chosen;
    if the slope is negative, the one with the largest x is chosen. For vertical lines
    (slope is None), the middle intersection is returned.

    Args:
        intersections (list[Intersection]): List of intersection objects.
        local_line (Line): Line object with a `slope` attribute.

    Returns:
        Intersection | None: Selected intersection point, or None if no intersections exist.
    """
    if not intersections:
        return None

    def x_of(inter: Intersection) -> int:
        p = inter.point
        return p.x if hasattr(p, "x") else p[0]

    intersections.sort(key=x_of)

    slope = getattr(local_line, "slope", None)
    if slope is None:
        return intersections[len(intersections) // 2]

    return intersections[0] if slope > 0 else intersections[-1]


def get_further_outer_baseline_corner(
    img: np.ndarray,
    local_line: Line,
    cannys_thresh_lower: int,
    cannys_thresh_upper: int,
    hough_thresh: int = 10,
    min_line_len: int = 10,
    max_line_gap: int = 10,
) -> Intersection | None:
    """
    Detects and returns the outer baseline corner intersection furthest along the local line.

    The function applies Canny edge detection, fills edge gaps, detects lines using the
    probabilistic Hough transform, groups them by slope, and finds valid intersections
    forming acute (< 90°) angles with opposite slopes. It filters out irrelevant intersections
    based on pixel patterns along the analyzed line and selects the furthest valid corner
    using the line’s slope direction.

    Args:
        img (np.ndarray): RGB input image.
        local_line (Line): Reference line used to determine direction and slope.
        cannys_thresh_lower (int): Lower threshold for Canny edge detection.
        cannys_thresh_upper (int): Upper threshold for Canny edge detection.
        hough_thresh (int, optional): Threshold for Hough line detection (default: 10).
        min_line_len (int, optional): Minimum line segment length (default: 10).
        max_line_gap (int, optional): Maximum allowed gap between line segments (default: 10).

    Returns:
        Intersection | None: The detected outer baseline corner intersection, or None if not found.
    """
    img_piece_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_piece_gray, cannys_thresh_lower, cannys_thresh_upper)
    filled_edges = fill_edges_image(edges)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap
    )
    if lines is None:
        lines = []

    line_obj = [Line.from_hough_line(line[0]) for line in lines]
    line_obj = [line for line in line_obj if line.slope is not None]
    grouped_lines = group_lines(line_obj)

    img_copy = img.copy()
    for line in grouped_lines:
        pts = line.limit_to_img(img_copy)
        cv2.line(img_copy, *pts, (0, 255, 0), 1)

    SETTINGS.debug and _plot_objs(edges, filled_edges, img_copy)

    if check_items_sign(grouped_lines):
        return None

    intersections = []
    for line_outer in grouped_lines:
        for line_inner in grouped_lines:
            if line_outer is line_inner:
                continue

            if np.sign(line_outer.slope) == np.sign(line_inner.slope):
                continue

            if angle_between_lines(line_outer, line_inner) >= 90:
                continue

            intersection = line_outer.intersection(line_inner, img)
            if intersection is None:
                continue

            col_start = intersection.point.x
            if local_line.slope > 0:
                col_range = range(0, col_start)
            else:
                col_range = range(col_start, edges.shape[1])

            skip_line = False

            sequence = []
            if np.sign(intersection.line1.slope) == np.sign(local_line.slope):
                analyze_line = intersection.line1
            else:
                analyze_line = intersection.line2

            for col in col_range:
                row = analyze_line.y_for_x(col)
                if row >= 0:
                    seq_num = int(filled_edges[row, col])

                    if not sequence or sequence[-1] != seq_num:
                        sequence.append(seq_num)
                        sequence.append(seq_num)

            if not (
                all(x == 0 for x in sequence)
                or (local_line.slope > 0 and sequence[0] == 0 and sequence[-1] == 1)
                or (local_line.slope < 0 and sequence[0] == 1 and sequence[-1] == 0)
            ):
                skip_line = True

            if skip_line:
                continue

            if intersection not in intersections:
                intersections.append(intersection)

    return _select_intersection_by_x(intersections, local_line)


def get_closest_line(lines: list[Line], point: Point) -> Line:
    """
    Returns the line closest to a given point.

    The function computes the perpendicular distance between the point and each
    line (handling both vertical and non-vertical cases) and returns the line
    with the smallest distance.

    Args:
        lines (list[Line]): List of line objects with `slope`, `intercept`, and optionally `xv` for vertical lines.
        point (Point): Point object or tuple (x, y) representing the reference point.

    Returns:
        Line: The line closest to the given point.
    """
    x, y = point
    min_dist = float("inf")
    closest = None

    for line in lines:
        if line.xv is not None:
            dist = abs(x - line.xv)
        else:
            dist = abs(line.slope * x - y + line.intercept) / np.sqrt(line.slope**2 + 1)

        if dist < min_dist:
            min_dist = dist
            closest = line

    return closest


def find_point_neighbourhood_simple(
    point: Point, size: int, img: np.ndarray, local_line: Line
) -> tuple[np.ndarray, int, int]:
    """
    Extract a rectangular neighborhood around a given point, positioning the point
    based on the local line's slope.

    Args:
        point (Point): The central point from which the neighborhood is calculated.
        size (int): Half-size of the neighborhood (creates size*2 x size*2 window).
        img (np.ndarray): The source image array.
        local_line (Line): Line used to determine point positioning within the window.
                          - Positive slope: point positioned near left boundary
                          - Negative slope: point positioned near right boundary

    Returns:
        tuple[np.ndarray, int, int]:
            - The extracted sub-image as a numpy array
            - The x-coordinate of the top-left corner (global coordinates)
            - The y-coordinate of the top-left corner (global coordinates)
    """
    height, width = img.shape[0], img.shape[1]

    center_x = int(point.x)
    center_y = int(point.y)

    if local_line.slope is not None:
        if local_line.slope > 0:
            x_start = max(0, center_x - size // 4)
            x_end = min(width - 1, x_start + 2 * size)
        else:
            x_end = min(width - 1, center_x + size // 4)
            x_start = max(0, x_end - 2 * size)
    else:
        x_start = max(0, center_x - size)
        x_end = min(width - 1, center_x + size)

    y_start = max(0, center_y - size)
    y_end = min(height - 1, center_y + size)

    if x_start < 0:
        x_end = min(width - 1, x_end - x_start)
        x_start = 0
    if x_end >= width:
        x_start = max(0, x_start - (x_end - width + 1))
        x_end = width - 1

    if y_start < 0:
        y_end = min(height - 1, y_end - y_start)
        y_start = 0
    if y_end >= height:
        y_start = max(0, y_start - (y_end - height + 1))
        y_end = height - 1

    return img[y_start : y_end + 1, x_start : x_end + 1], x_start, y_start


def find_point_neighbourhood_simple_no_line(point: Point, size: int, img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Extracts a square neighborhood around a point within image bounds.

    The function returns a cropped region of the image centered on the given point,
    constrained by the image dimensions, along with the top-left coordinates of the
    extracted region.

    Args:
        point (Point): Center point around which the neighborhood is extracted.
        size (int): Half-size of the square neighborhood (in pixels).
        img (np.ndarray): Input image array.

    Returns:
        tuple[np.ndarray, int, int]: Cropped image region, starting x-coordinate, and starting y-coordinate.
    """
    height, width = img.shape[0], img.shape[1]

    point = point.as_int()

    x_start = max(point.x - size, 0)
    y_start = max(point.y - size, 0)

    x_end = min(point.x + size, width - 1)
    y_end = min(point.y + size, height - 1)

    return img[y_start : y_end + 1, x_start : x_end + 1], x_start, y_start


def crop_court_field(
    image: np.ndarray,
    baseline: Line,
    closer_outer_baseline_point: Point,
    closer_outer_netline_point: Point,
    further_outer_baseline_point: Point,
    further_outer_netline_point: Point,
) -> np.ndarray:
    """
    Crops the court field region from the image based on baseline orientation and key boundary points.

    The function determines the cropping rectangle from the provided baseline and
    outer boundary points (baseline and netline intersections) and returns the cropped
    court field area along with its top-left coordinates.

    Args:
        image (np.ndarray): Input RGB image containing the court.
        baseline (Line): Reference baseline line used to determine crop direction.
        closer_outer_baseline_point (Point): Closest outer point on the baseline.
        closer_outer_netline_point (Point): Closest outer point on the netline.
        further_outer_baseline_point (Point): Farthest outer point on the baseline.
        further_outer_netline_point (Point): Farthest outer point on the netline.

    Returns:
        tuple[np.ndarray, int, int]: Cropped court field image, starting x-coordinate, and starting y-coordinate.
    """
    x_start, x_end, y_start, y_end = {
        True: (
            further_outer_netline_point.x,
            closer_outer_netline_point.x + 1,
            further_outer_netline_point.y,
            closer_outer_baseline_point.y + 1,
        ),
        False: (
            closer_outer_netline_point.x,
            further_outer_baseline_point.x + 1,
            further_outer_netline_point.y,
            closer_outer_baseline_point.y + 1,
        ),
    }[baseline.slope > 0]

    return image[y_start:y_end, x_start:x_end], x_start, y_start


def image_to_lines(
    image: np.ndarray,
    bin_thresh: float,
    cannys_lower_thresh: int,
    cannys_upper_thresh: int,
    hough_thresh: int,
    min_line_len: float,
    hough_max_line_gap: int,
    reference_line: Line,
    same_slope_sign: bool = False,
) -> list[LineGroup]:
    """
    Detects and groups lines from an image based on their slope relative to a reference line.

    The function binarizes and edge-detects the image, extracts lines using the
    probabilistic Hough transform, filters them by slope direction relative to
    the reference line, and groups similar lines together.

    Args:
        image (np.ndarray): RGB input image.
        bin_thresh (float): Binary threshold ratio for image binarization.
        cannys_lower_thresh (int): Lower threshold for Canny edge detection.
        cannys_upper_thresh (int): Upper threshold for Canny edge detection.
        hough_thresh (int): Threshold for Hough line detection.
        min_line_len (float): Minimum line segment length.
        hough_max_line_gap (int): Maximum allowed gap between line segments.
        reference_line (Line): Line used for slope comparison.
        same_slope_sign (bool, optional): If True, keeps lines with the same slope sign as the reference;
                                          otherwise keeps lines with the opposite sign (default: False).

    Returns:
        list[LineGroup]: Grouped line objects filtered and organized by slope similarity.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin_img = (img_gray > img_gray.max() * bin_thresh).astype(np.uint8) * 255
    edges = cv2.Canny(bin_img, cannys_lower_thresh, cannys_upper_thresh)

    SETTINGS.debug and _plot_objs(edges)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=hough_max_line_gap
    )
    lines = [] if lines is None else lines

    ref_sign = np.sign(reference_line.slope)
    cmp = (lambda a, b: a == b) if same_slope_sign else (lambda a, b: a != b)

    line_obj = [
        line
        for line in (Line.from_hough_line(line_data[0]) for line_data in lines)
        if line.slope is not None and cmp(np.sign(line.slope), ref_sign)
    ]

    return group_lines(line_obj)


def detect_lines_opposite_slope(
    image: np.ndarray,
    canny_lower: float,
    canny_upper: float,
    *,
    hough_threshold: int,
    min_line_length: float,
    max_line_gap: float,
    ref_line: Line,
    rho: float = 1.0,
    theta: float = np.pi / 180,
) -> list[Line]:
    """
    Run Canny + HoughLinesP, convert to Line objects, and keep only lines whose
    slope sign differs from ref_line.slope.

    Returns:
        list[Line]: filtered Line objects
    """
    edges = cv2.Canny(image, canny_lower, canny_upper)
    raw = cv2.HoughLinesP(
        edges,
        rho,
        theta,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    raw = [] if raw is None else raw

    line_objs = [Line.from_hough_line(line[0]) for line in raw]
    ref_sign = np.sign(ref_line.slope)
    return [ln for ln in line_objs if (ln.slope is not None) and (np.sign(ln.slope) != ref_sign)]


def create_reference_court(
    ref_img_height: int = 25_000, ref_img_width: int = 11_000, line_thickness: int = 50
) -> tuple[dict[str, Point], np.ndarray]:
    """
    Creates a synthetic reference tennis court image and returns its key reference points.

    The function draws outer/inner sidelines, baselines, service lines, and the center/net
    lines on a blank image using predefined metric constants, and returns a dictionary of
    named reference points (as `Point`) along with the rendered RGB image.

    Args:
        ref_img_height (int, optional): Output image height in pixels (default: 25_000).
        ref_img_width (int, optional): Output image width in pixels (default: 11_000).
        line_thickness (int, optional): Line thickness used for drawing (default: 50).

    Returns:
        tuple[dict[str, Point], np.ndarray]:
            - dict[str, Point]: Named reference points:
                - "closer_outer_netline_point", "closer_outer_baseline_point",
                  "further_outer_netline_point", "further_outer_baseline_point",
                  "closer_inner_netline_point", "closer_inner_baseline_point",
                  "further_inner_netline_point", "further_inner_baseline_point",
                  "closer_service_point", "further_service_point",
                  "net_service_point", "centre_service_point".
            - np.ndarray: The rendered reference court image (RGB).
    """
    ref_closer_outer_netline_point = 0, COURT_LENGTH_HALF
    ref_closer_outer_baseline_point = 0, LENGTH
    ref_closer_outer_baseline_point_2 = 0, 0

    ref_further_outer_netline_point = WIDTH, COURT_LENGTH_HALF
    ref_further_outer_baseline_point = WIDTH, LENGTH
    ref_further_outer_baseline_point_2 = WIDTH, 0

    ref_closer_inner_netline_point = DIST_OUTER_SIDELINE, COURT_LENGTH_HALF
    ref_closer_inner_baseline_point = DIST_OUTER_SIDELINE, LENGTH
    ref_closer_inner_baseline_point_2 = DIST_OUTER_SIDELINE, 0

    ref_further_inner_netline_point = WIDTH - DIST_OUTER_SIDELINE, COURT_LENGTH_HALF
    ref_further_inner_baseline_point = WIDTH - DIST_OUTER_SIDELINE, LENGTH
    ref_further_inner_baseline_point_2 = WIDTH - DIST_OUTER_SIDELINE, 0

    ref_closer_service_point = DIST_OUTER_SIDELINE, LENGTH - DIST_FROM_BASELINE
    ref_further_service_point = WIDTH - DIST_OUTER_SIDELINE, LENGTH - DIST_FROM_BASELINE

    ref_closer_service_point_2 = DIST_OUTER_SIDELINE, DIST_FROM_BASELINE
    ref_further_service_point_2 = WIDTH - DIST_OUTER_SIDELINE, DIST_FROM_BASELINE

    ref_net_service_point = COURT_WIDTH_HALF, COURT_LENGTH_HALF
    ref_centre_service_point = COURT_WIDTH_HALF, LENGTH - DIST_FROM_BASELINE
    ref_centre_service_point_2 = COURT_WIDTH_HALF, DIST_FROM_BASELINE

    ref_img = np.zeros((ref_img_height, ref_img_width, 3), np.uint8)

    cv2.line(ref_img, ref_closer_outer_baseline_point, ref_closer_outer_netline_point, (255, 0, 0), line_thickness)
    cv2.line(ref_img, ref_closer_outer_netline_point, ref_closer_outer_baseline_point_2, (0, 255, 0), line_thickness)

    cv2.line(ref_img, ref_further_outer_baseline_point, ref_further_outer_netline_point, (255, 0, 0), line_thickness)
    cv2.line(ref_img, ref_further_outer_netline_point, ref_further_outer_baseline_point_2, (0, 255, 0), line_thickness)

    cv2.line(ref_img, ref_closer_outer_baseline_point, ref_further_outer_baseline_point, (255, 0, 0), line_thickness)
    cv2.line(
        ref_img, ref_closer_outer_baseline_point_2, ref_further_outer_baseline_point_2, (0, 255, 0), line_thickness
    )

    cv2.line(ref_img, ref_closer_inner_baseline_point, ref_closer_inner_netline_point, (255, 0, 0), line_thickness)
    cv2.line(ref_img, ref_closer_inner_netline_point, ref_closer_inner_baseline_point_2, (0, 255, 0), line_thickness)

    cv2.line(ref_img, ref_further_inner_baseline_point, ref_further_inner_netline_point, (255, 0, 0), line_thickness)
    cv2.line(ref_img, ref_further_inner_netline_point, ref_further_inner_baseline_point_2, (0, 255, 0), line_thickness)

    cv2.line(ref_img, ref_closer_service_point, ref_further_service_point, (255, 0, 0), line_thickness)
    cv2.line(ref_img, ref_closer_service_point_2, ref_further_service_point_2, (0, 255, 0), line_thickness)

    cv2.line(ref_img, ref_net_service_point, ref_centre_service_point, (255, 0, 0), line_thickness)
    cv2.line(ref_img, ref_net_service_point, ref_centre_service_point_2, (0, 255, 0), line_thickness)

    cv2.line(ref_img, ref_closer_outer_netline_point, ref_further_outer_netline_point, (255, 0, 0), line_thickness)

    return {
        name: Point.from_iterable(point)
        for name, point in {
            "closer_outer_netline_point": ref_closer_outer_netline_point,
            "closer_outer_baseline_point": ref_closer_outer_baseline_point,
            "further_outer_netline_point": ref_further_outer_netline_point,
            "further_outer_baseline_point": ref_further_outer_baseline_point,
            "closer_inner_netline_point": ref_closer_inner_netline_point,
            "closer_inner_baseline_point": ref_closer_inner_baseline_point,
            "further_inner_netline_point": ref_further_inner_netline_point,
            "further_inner_baseline_point": ref_further_inner_baseline_point,
            "closer_service_point": ref_closer_service_point,
            "further_service_point": ref_further_service_point,
            "net_service_point": ref_net_service_point,
            "centre_service_point": ref_centre_service_point,
        }.items()
    }, ref_img


def warp_points(
    ref_points: dict[str, Point], dst_points: dict[str, Point], src_image: np.ndarray, ref_img: np.ndarray, *names: str
) -> tuple[dict[str, Point], np.ndarray, np.ndarray]:
    """
    Warps a reference court to match destination points and maps all reference points.

    Using the named point pairs (`names`) from `ref_points` → `dst_points`, the function
    estimates a homography, warps `ref_img` to the `src_image` frame, overlays the result,
    and projects every reference point through the homography.

    Args:
        ref_points (dict[str, Point]): Reference points in the canonical court.
        dst_points (dict[str, Point]): Corresponding target points in the source image.
        src_image (np.ndarray): Source image to align onto (defines output size).
        ref_img (np.ndarray): Reference court image to be warped.
        *names (str): Keys of point pairs to use for homography estimation.

    Returns:
        tuple[dict[str, Point], np.ndarray, np.ndarray]:
            - dict[str, Point]: All reference points transformed into source-image coordinates.
            - np.ndarray: Warped reference image (same size as `src_image`).
            - np.ndarray: Blended overlay (`src_image` + warped reference).
    """
    ref_points_arr = np.float32([ref_points[n].to_tuple() for n in names])
    dst_points_arr = np.float32([dst_points[n].to_tuple() for n in names])
    H, _ = cv2.findHomography(ref_points_arr, dst_points_arr)

    height, width = src_image.shape[:2]
    transformed_img = cv2.warpPerspective(ref_img, H, (width, height))

    covered_img = cv2.addWeighted(src_image, 1, transformed_img, 1, 0)

    all_ref_points_arr = np.array([(p.x, p.y) for p in ref_points.values()], dtype=np.float32)[np.newaxis, ::]
    transformed_points = cv2.perspectiveTransform(all_ref_points_arr, H)
    coords = transformed_points[0]
    transformed_points_dict = {name: Point(*coords[i]) for i, name in enumerate(ref_points.keys())}

    return transformed_points_dict, transformed_img, covered_img


def plot_results(img: np.ndarray, path: Path, pic_name: str, lines: dict[str, Line], points: dict[str, Point]) -> None:
    """
    Draws detected points and lines on an image and saves the visualization.

    The function overlays all provided points and lines on a copy of the input image,
    then saves the resulting visualization to the specified path.

    Args:
        img (np.ndarray): Input RGB image.
        path (Path): Directory path where the output image will be saved.
        pic_name (str): Output image filename.
        lines (dict[str, Line]): Dictionary of line objects to draw.
        points (dict[str, Point]): Dictionary of point objects to mark.
    """
    pic = img.copy()

    for point in points.values():
        cv2.circle(pic, point, 1, (255, 0, 0), 3, -1)

    for line in lines.values():
        pt = line.limit_to_img(pic)
        cv2.line(pic, *pt, (0, 0, 255))

    Image.fromarray(pic).save(path / pic_name)


def measure_error(
    img: np.ndarray,
    found_points: dict[str, Point],
    ground_truth_points: dict[str, dict[str, float]],
    prefix: str | None = None,
) -> dict[str, float]:
    """
    Computes Euclidean distance errors between detected and ground-truth points.

    Each ground-truth point (expressed as percentage coordinates) is transformed
    into pixel coordinates, and its distance to the corresponding detected point
    is calculated. Results are stored in a dictionary with optional prefixing.

    Args:
        img (np.ndarray): Image used to convert ground-truth coordinates to pixels.
        found_points (dict[str, Point]): Detected points keyed by name.
        ground_truth_points (dict[str, dict[str, float]]): Ground-truth points as normalized coordinates.
        prefix (str | None, optional): Optional prefix for output keys (default: None).

    Returns:
        dict[str, float]: Mapping of point names to distance errors in pixels.
    """
    errors = {}
    for name, pt in found_points.items():
        raw_gtpt = ground_truth_points.get(name)
        transformed_gtpt = transform_annotation(img, raw_gtpt)
        distance_error = transformed_gtpt.distance(pt)
        key = f"{prefix}_{name}_dist" if prefix else f"{name}_dist"
        errors.update({key: distance_error})
    return errors


def load_config[T: BaseModel](config_filepath: str | Path, data_model: type[T]) -> T:
    """
    Loads a JSON configuration file and validates it against a Pydantic model.

    The function reads and parses a JSON file, validates its structure and content
    using the specified Pydantic model, and returns the validated configuration.
    Supports both single-object (dict) and list-based (list of objects) JSON structures.

    Args:
        config_filepath (str | Path): Path to the JSON configuration file.
        data_model (type[BaseModel]): Pydantic model class used for validation.

    Returns:
        BaseModel: Validated configuration instance (or list of instances).

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the JSON structure or validation is invalid.
    """
    config_path = Path(config_filepath)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        config_data = json.load(file)

    try:
        if isinstance(config_data, dict):
            return data_model(**config_data)

        elif isinstance(config_data, list):
            adapter = TypeAdapter(list[data_model])
            return adapter.validate_python(config_data)

        else:
            raise ValueError(
                f"Invalid JSON type in {config_path}: expected dict or list, got {type(config_data).__name__}"
            )
    except ValidationError as e:
        raise ValueError(f"Invalid configuration in {config_path}: {e}") from e


def compose_court_data(
    closer_outer_baseline_point: Point,
    closer_outer_netline_point: Point,
    further_outer_baseline_point: Point,
    further_outer_netline_point: Point,
    closer_inner_baseline_point: Point,
    further_inner_baseline_point: Point,
    closer_inner_netline_point: Point,
    further_inner_netline_point: Point,
    net_service_point: Point,
    centre_service_point: Point,
    further_service_point: Point,
    closer_service_point: Point,
    closer_outer_sideline: Line,
    baseline: Line,
    netline: Line,
    further_outer_sideline: Line,
    closer_inner_sideline: Line,
    further_inner_sideline: Line,
    centre_service_line: Line,
    service_line: Line,
    data: GroundTruthCourtPoints = None,
) -> tuple[dict[str, Point], dict[str, Line], dict[str, dict[str, float]]]:
    """
    Compose the destination and ground truth court geometry dictionaries.
    ground_truth_points are transformed into {"x": ..., "y": ...} format.

    Returns:
        tuple: (dst_points, dst_lines, ground_truth_points)
    """

    dst_points = {
        "closer_outer_baseline_point": closer_outer_baseline_point,
        "closer_outer_netline_point": closer_outer_netline_point,
        "further_outer_baseline_point": further_outer_baseline_point,
        "further_outer_netline_point": further_outer_netline_point,
        "closer_inner_baseline_point": closer_inner_baseline_point,
        "further_inner_baseline_point": further_inner_baseline_point,
        "closer_inner_netline_point": closer_inner_netline_point,
        "further_inner_netline_point": further_inner_netline_point,
        "net_service_point": net_service_point,
        "centre_service_point": centre_service_point,
        "further_service_point": further_service_point,
        "closer_service_point": closer_service_point,
    }

    dst_lines = {
        "closer_outer_sideline": closer_outer_sideline,
        "baseline": baseline,
        "netline": netline,
        "further_outer_sideline": further_outer_sideline,
        "closer_inner_sideline": closer_inner_sideline,
        "further_inner_sideline": further_inner_sideline,
        "centre_service_line": centre_service_line,
        "service_line": service_line,
    }

    if data is None:
        return dst_points, dst_lines

    ground_truth_points = data.ground_truth_points.model_dump()

    return dst_points, dst_lines, ground_truth_points


def validate_data_and_pictures(config_data: list, pictures: list[str]) -> None:
    """
    Validates that config_data and pictures have the same length.
    Raises a ValueError if they don't match.

    Args:
        config_data (list): The data entries from the config file.
        pictures (list): The list of loaded pictures.

    Raises:
        ValueError: If the lengths don't match.
    """
    if len(config_data) != len(pictures):
        msg = f"Length mismatch between config.data ({len(config_data)}) and pictures ({len(pictures)})."

        raise ValueError(msg)


def get_point_weights(row: pd.Series) -> float:
    """
    Calculates a weighted average error for a set of court points.

    The function applies predefined weights to each point’s distance error
    (columns ending with "_dist") and returns the weighted mean, giving
    greater importance to specific key points.

    Args:
        row (pd.Series): Row containing distance errors for multiple points,
                         with column names following the pattern "<point_name>_dist".

    Returns:
        float: Weighted average of distance errors across all valid points.
    """
    WEIGHTS = {
        "closer_outer_baseline_point": 1,
        "closer_inner_baseline_point": 2,
        "closer_outer_netpoint": 3,
        "further_outer_baseline_point": 4,
        "further_inner_baseline_point": 5,
        "closer_inner_netpoint": 6,
        "further_outer_netpoint": 7,
        "further_inner_netpoint": 7,
    }

    weighted_sum = 0
    applied_weights = []
    for col, val in row.items():
        point_name = col.replace("_dist", "")
        if point_name in WEIGHTS.keys():
            weight = WEIGHTS.get(point_name)
            weighted_sum += val * weight
            applied_weights.append(weight)

    return weighted_sum / sum(applied_weights)


def draw_and_display(
    img: np.ndarray,
    *,
    lines: Optional[Iterable[Line]] = None,
    points: Optional[Iterable[Point]] = None,
) -> None:
    """
    Draw lines and points on an image and display it using a given plotting function.

    Handles both raw point pairs and objects with a `limit_to_img(img)` method.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    lines : Iterable[LineInput], optional
        Lines to draw. Each item can be a tuple of points ((x1, y1), (x2, y2))
        or an object exposing `.limit_to_img(img) -> ((x1, y1), (x2, y2))`.
    points : Iterable[Point], optional
        Points to draw as small filled circles (x, y).
    """
    img_copy = img.copy()
    if lines:
        for line in lines:
            pts = line.limit_to_img(img_copy)
            cv2.line(img_copy, *pts, (255, 0, 0), 1)

    if points:
        for p in points:
            cv2.circle(img_copy, p, 1, (255, 0, 0), -1)

    _plot_objs(img_copy)
