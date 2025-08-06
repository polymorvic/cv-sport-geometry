import os
import cv2
import numpy as np


def get_pictures(path: str) -> dict[str, list[np.ndarray]]:
    """Loads all valid image files from a directory and returns their representations in RGB, HSV, and grayscale formats.
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
    """It applies probabilistic Hough line transformation to given RGB image
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

