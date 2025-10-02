import os
import cv2
import random
import colorsys
import numpy as np
from skimage.morphology import skeletonize
from typing import Literal, Iterable, Optional
from utils.lines import Line, LineGroup, Point, Intersection
from utils.const import ARRAY_X_INDEX, ARRAY_Y_INDEX
import matplotlib.pyplot as plt


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
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    filenames = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in valid_extensions])

    pics = {
        'rgb': [],
        'hsv': [],
        'gray': []
    }

    for filename in filenames:
        img_path = os.path.join(path, filename)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            continue

        pics['rgb'].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pics['hsv'].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
        pics['gray'].append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    return pics


def apply_hough_transformation(img_rgb: np.ndarray, blur_kernel_size: int = 5, canny_thresh_lower: int = 50, canny_thresh_upper: int = 150, hough_thresh: int = 100, hough_min_line_len: int = 100, hough_max_line_gap: int = 10) -> tuple[np.ndarray, list]:
    """
    It applies probabilistic Hough line transformation to given RGB image
    process of applying hough transformation to the image is as follows:
        - creating copy of original image,
        - then copy is converted to gray scale, 
        - later the gaussian blur filter is applied to the gray scaled image
        - on the blurred image Canny's edges detection is processed
        - on the binary image that contains the detected edges Hough transformation is processed to get lines,
            if any detected, it contains two points - both ends of line, between them we can plot strainght line

    In HoughLinesP function - constant values of 1 and np.pi/180 indicates respectively rho and theta parameters that dont need to be tuned

    Args:
        img_rgb (np.ndarray): an RGB image
        blur_kernel_size (int, optional): size of square blur kernel. Defaults to 5.
        canny_thresh_lower (int, optional): lower threshold for the hysteresis process in Canny. Defaults to 50.
        canny_thresh_upper (int, optional): upper thresholds for the hysteresis process in Canny. Defaults to 150.
        hough_thresh (int, optional): minimum number of intersections (votes) in the accumulator to "declare" a line - the higher value the fewer lines, only strong ones. Defaults to 100.
        hough_min_line_len (int, optional): the minimum length (in pixels) of a line segment to be accepted, short segments below this length are ignored. Defaults to 100.
        hough_max_line_gap (int, optional): the maximum allowed gap between two line segments to treat them as a single line, if endpoints of two segments are close enough (within this gap), they are joined into one line. Defaults to 10.

    Returns:
        tuple[np.ndarray, list]: image with detected lines drawn, list of lists of 4 integers each item - that indicates both ends of detected lines 
    """
    img_copy = img_rgb.copy()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_thresh_lower, canny_thresh_upper)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh, minLineLength=hough_min_line_len, maxLineGap=hough_max_line_gap)
    if lines is None:
        lines = []
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_copy, lines


def group_lines(lines: list[Line], thresh_theta: float | int = 5, thresh_intercept: float | int = 10) -> list[LineGroup]:
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
            # No group matched, create a new group
            groups.append(LineGroup([line]))

    return groups


def draw_line_group(img: np.ndarray, line_group: LineGroup, color: tuple[int, int, int], approx_color: tuple[int, int, int], approx_only: bool = True) -> np.ndarray:
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

        # convert to bgr for opencv
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
            start -= (end - (limit - 1))
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

        # print(f'{x_start=}{x_end=}')
        if x_start > x_end:
            x_start, x_end = x_end, x_start

        # print(f'{x_start=}{x_end=}')

        # print(f'{line.slope=}{line.intercept=}')
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

        # print(f'{y_start=}{y_end=}')

        if y_start > y_end:
            y_start, y_end = y_end, y_start

        # print(f'{y_start=}{y_end=}')

        # print('!!!!!!!!!!!!!!!!', f'{y_start=}{y_end=}', f'{x_start=}{x_end=}', '!!!!!!!!!!!!!!!!')

    return img[y_start:y_end+1, x_start:x_end+1], x_start, y_start


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
    norm = (dx*dx + dy*dy) ** 0.5
    if norm == 0:
        return (0.0, -1.0) 
    return (dx / norm, dy / norm)


def traverse_line(point: Point, offset: int, img: np.ndarray, line: Line, direction: Literal["up", "down"] = "up", neighbourhood_type: Literal['complex', 'simple'] = 'complex') -> tuple[Point, np.ndarray, list[int, int]]:
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

    neighbourhood_func = {
        'simple': find_point_neighbourhood_simple,
        'complex': find_point_neighbourhood
    }[neighbourhood_type]

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


def find_net_lines(img_piece: np.ndarray, cannys_thresh_lower: int = 50, cannys_thresh_upper: int = 150, hough_thresh: int = 10, min_line_len: int = 10, max_line_gap:int = 10):
    piece_gray = cv2.cvtColor(img_piece, cv2.COLOR_RGB2GRAY)
    neg_gray_img = 255 - piece_gray
    edges = cv2.Canny(neg_gray_img, cannys_thresh_lower, cannys_thresh_upper)

    # plt.imshow(neg_gray_img)
    # plt.show()

    # plt.imshow(edges)
    # plt.show()

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        lines = []
    
    img_copy = img_piece.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # plt.imshow(img_copy)
    # plt.show()

    line_obj = [Line.from_hough_line(line[0]) for line in lines]
    line_obj = [line for line in line_obj if line.slope is not None] 
    return group_lines(line_obj)


def check_items_sign(line_groups: list[LineGroup]) -> bool:
    return all(item.slope > 0 for item in line_groups) or all(item.slope < 0 for item in line_groups)


def transform_point(point: Intersection | Point, original_x_start: int, original_y_start: int, to_global: bool = True) -> Point:
    if isinstance(point, Intersection):
        point = point.point

    if to_global:
        return Point(point.x + original_x_start, point.y + original_y_start)
    else:
        return Point(point.x - original_x_start, point.y - original_y_start)


def transform_line(original_line: Line, original_img: np.ndarray, original_x_start: int, original_y_start: int, to_global: bool = True) -> Line:
    pts_source: Iterable[Point] = original_line.limit_to_img(original_img)
    pts_transformed = [transform_point(p, original_x_start, original_y_start, to_global=to_global) for p in pts_source]
    return Line.from_points(*pts_transformed)


def transform_intersection(intersection: Intersection, source_img: np.ndarray, original_x_start: int, original_y_start: int, to_global: bool = True) -> Intersection:
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
    counter = 0
    for i, item in enumerate(arr):

        if i > 0:
            if item - arr[i-1] > 1:
                counter += 1
        else:
            counter += 1

    return counter


def is_court_corner(img: np.ndarray, intersect_point: Point, original_range: tuple[int, int], bin_thresh: float = 0.8, x_range: int = 3, y_range: int = 3) -> bool:

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bin_img = (gray > gray.max() * bin_thresh).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed_bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(img)
    # plt.show()

    # plt.imshow(gray)
    # plt.show()

    # plt.imshow(bin_img)
    # plt.show()

    local_intersect_point = transform_point(intersect_point, *original_range, False)
    if np.sum(closed_bin_img[local_intersect_point.y - y_range: local_intersect_point.y + y_range, local_intersect_point.x - x_range: local_intersect_point.x + x_range]) == 0:
        return False
    
    # cv2.circle(img, local_intersect_point, 2, (0, 255, 0))
    # plt.imshow(img)
    # plt.show()


    ones_iloc = np.argwhere(closed_bin_img > 0)
    x_range = np.unique(ones_iloc[:,1])
    y_range = np.unique(ones_iloc[:,0])

    if len(x_range) == 0 or len(x_range) == closed_bin_img.shape[1] and len(y_range) == closed_bin_img.shape[0]:
        # print('uniques')
        return False

    if not np.all(np.diff(x_range)==1):
        return False
    
    row_start, row_stop = ones_iloc[:,0].min(), ones_iloc[:,0].max()
    seq_groups = []
    for row in range(row_start, row_stop + 1):
        ones = np.argwhere(closed_bin_img[row, :]).flatten()
        seq_num = _count_array_sequence_group(ones)

        # print(f'{seq_num}-->{ones}')

        if not seq_groups or seq_groups[-1] != seq_num:
            seq_groups.append(seq_num)

    # print('-->'.join(map(str, seq_groups)))

    if seq_groups != [1, 2, 1] and seq_groups != [2, 1]:
        return False
    
    # print('return true')
    return True


def angle_between_lines(line1: Line, line2: Line) -> float | None:
    """
    Returns the smallest angle in degrees between two lines.
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


def is_inner_sideline(img: np.ndarray, bin_thresh: float = 0.8, hough_line_thresh: int = 8, min_line_len: int | None = 5 , min_line_gap: int = 5) -> bool:

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bin_img = (gray > gray.max() * bin_thresh).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # closed_bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    skel = skeletonize(bin_img).astype(np.uint8)

    # plt.imshow(img)
    # plt.show()

    # plt.imshow(gray)
    # plt.show()

    # plt.imshow(bin_img)
    # plt.show()

    # plt.imshow(skel)
    # plt.show()

    lines = cv2.HoughLinesP(skel, 1, np.pi/180, threshold=hough_line_thresh, minLineLength=min_line_len, maxLineGap=min_line_gap)
    if lines is None:
        lines = []
            
    img_copy = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # print(len(lines))
    # plt.imshow(img_copy)
    # plt.show()  

    line_obj = [Line.from_hough_line(line[0]) for line in lines]
    line_obj = [line for line in line_obj if line.slope is not None] 
    line_groups = group_lines(line_obj)

    if len(set(np.sign(line.slope) for line in line_groups)) <= 1:
        return False
    
    angle = angle_between_lines(line_groups[0], line_groups[1])

    return angle is not None and angle < 90


def transform_annotation(img: np.ndarray, annotation: dict[Literal['x', 'y'], float]) -> Point:
    height, width = img.shape[:2]
    x = annotation['x'] / 100 * width
    y = annotation['y'] / 100 * height
    return Point(x, y)


def fill_edges_image(edges_img: np.ndarray):
    h, w = edges_img.shape
    filled = np.zeros_like(edges_img)

    for x in range(w):
        ys = np.flatnonzero(edges_img[:, x])
        if ys.size >= 2:
            y1, y2 = ys.min(), ys.max()
            filled[y1:y2 + 1, x] = 1 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)


def _select_intersection_by_x(intersections: list[Intersection], local_line: Line) -> Intersection | None:
    """Pick intersection by x: min x if local line slope > 0, else max x."""
    if not intersections:
        return None

    def x_of(inter: Intersection):
        p = inter.point
        return p.x if hasattr(p, "x") else p[0]

    intersections.sort(key=x_of)

    slope = getattr(local_line, "slope", None)
    if slope is None:
        return intersections[len(intersections) // 2]

    return intersections[0] if slope > 0 else intersections[-1]


def get_further_outer_baseline_corner(img: np.ndarray, local_line: Line, cannys_thresh_lower: int, cannys_thresh_upper: int, hough_thresh: int = 10, min_line_len: int = 10, max_line_gap: int = 10) -> Intersection:

    img_piece_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_piece_gray, cannys_thresh_lower, cannys_thresh_upper)
    filled_edges = fill_edges_image(edges)

    # plt.imshow(edges)
    # plt.show()

    # plt.imshow(filled_edges)
    # plt.show()

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        lines = []

    line_obj = [Line.from_hough_line(line[0]) for line in lines]
    line_obj = [line for line in line_obj if line.slope is not None] 
    grouped_lines = group_lines(line_obj)

    img_copy = img.copy()
    for line in grouped_lines:
        pts = line.limit_to_img(img_copy)
        cv2.line(img_copy, *pts, (0, 255, 0), 1)

    # plt.imshow(img_copy)
    # plt.show()

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
    
            # sums = []
            # for col in col_range:
            #     sums.append(np.sum(edges[:, col]))
            
            # print(sums)
            # if all(sums):
            #     skip_line = True

            sequence = []
            analyze_line = intersection.line1 if np.sign(intersection.line1.slope) == np.sign(local_line.slope) else intersection.line2
            for col in col_range:
                row = analyze_line.y_for_x(col)
                if row >= 0:
                    # print(f'{row=} {col=} {filled_edges[row, col]}')
                    seq_num = int(filled_edges[row, col])

                    if not sequence or sequence[-1] != seq_num:
                        sequence.append(seq_num)
                        sequence.append(seq_num)

            # print(sequence)
            # print('cond 1', all(x == 0 for x in sequence))
            # print('cond 2', local_line.slope > 0 and sequence[0] == 0 and sequence[-1] == 1)
            # print('cond 3', local_line.slope < 0 and sequence[0] == 1 and sequence[-1] == 0)

            if not (all(x == 0 for x in sequence) or (local_line.slope > 0 and sequence[0] == 0 and sequence[-1] == 1) or (local_line.slope < 0 and sequence[0] == 1 and sequence[-1] == 0)):
                skip_line = True

            if skip_line:
                continue

            if intersection not in intersections:
                intersections.append(intersection)

    return _select_intersection_by_x(intersections, local_line) # intersections[0] if len(intersections) > 0 else intersections


def get_closest_line(lines, point):
    """Find the line closest to a point."""
    x, y = point
    min_dist = float('inf')
    closest = None
    
    for line in lines:
        if line.xv is not None:
            # Vertical line
            dist = abs(x - line.xv)
        else:
            # Regular line: y = mx + b -> mx - y + b = 0
            dist = abs(line.slope * x - y + line.intercept) / np.sqrt(line.slope**2 + 1)
        
        if dist < min_dist:
            min_dist = dist
            closest = line
    
    return closest


def find_point_neighbourhood_simple(point: Point, size: int, img: np.ndarray, 
                                   local_line: Line) -> tuple[np.ndarray, int, int]:
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
    
    return img[y_start:y_end+1, x_start:x_end+1], x_start, y_start


def find_point_neighbourhood_simple_no_line(point: Point, size: int, img: np.ndarray) -> tuple[np.ndarray, int, int]:

    height, width = img.shape[0], img.shape[1]
    
    point = point.as_int()

    x_start = max(point.x - size, 0)
    y_start = max(point.y - size, 0)

    x_end = min(point.x + size, width - 1)
    y_end = min(point.y + size, height - 1)
    
    return img[y_start:y_end+1, x_start:x_end+1], x_start, y_start
