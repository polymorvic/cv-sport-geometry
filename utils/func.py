import os
import cv2
import random
import colorsys
import numpy as np
from utils.lines import Line, LineGroup, Point
from utils.const import ARRAY_X_INDEX, ARRAY_Y_INDEX

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
    valid_extensions = {'.jpg', '.jpeg', '.png',}
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

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


def group_lines(lines: list[Line], thresh_theta: float, thresh_intercept: float) -> list[LineGroup]:
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

    return img[y_start:y_end+1, x_start:x_end+1], x_start, y_start

