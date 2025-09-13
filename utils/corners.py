import numpy as np
from collections import defaultdict
from .lines import Line, Intersection, Point
from .func import (traverse_line, find_net_lines, check_items_sign, transform_intersection, transform_line,
                   is_court_corner, find_point_neighbourhood, is_inner_sideline)

import matplotlib.pyplot as plt
import cv2

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


    def __init__(self, intersections: list[Intersection], img: np.ndarray, offset: int = 20) -> None:
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
        self.offset = offset
        self.intersections = sorted(intersections, key = lambda intersection: -intersection.point.y)
        self.line_intersection_mapping: dict[Line, set[Intersection]] = defaultdict(set)

        for intersect in self.intersections:
            for line in (intersect.line1, intersect.line2):
                self.line_intersection_mapping[line].add(intersect)

    
    def find_closer_outer_baseline_point(self, bin_thresh: float) -> tuple[Intersection, Point]:
        """
        Finds the nearest outer baseline intersection point within a specific angular range.

        This method iterates through all stored intersections, filtering them by angle 
        (between 90 and 270 degrees). For each matching intersection, it searches along 
        both connected lines to find the nearest intersection that also matches the angle 
        condition. Once found, it traverses the corresponding line to locate the 
        corresponding "net-side" intersection point.

        Returns:
            tuple[Intersection, Point]: 
                - The original qualifying intersection.
                - The corresponding point found by traversing toward the net.
        
        Notes:
            - The `Intersection` objects are assumed to have `angle`, `line1`, `line2`, 
            and `point` attributes.
            - The method relies on `self.line_intersection_mapping` for connected intersections.
            - Returns `None` implicitly if no qualifying pair is found.
        """
        for intersect in self.intersections:

            if not 270 > intersect.angle > 90:
                continue

            nearest_intersection = None
            for line in (intersect.line1, intersect.line2):
                sorted_intersection_points = sorted(self.line_intersection_mapping[line], key = lambda intersection: intersection.distance(intersect))

                for inner_intersect in sorted_intersection_points:

                    if not 270 > inner_intersect.angle > 90:
                        continue
                    else:
                        nearest_intersection = inner_intersect
                        break

                if nearest_intersection is not None:

                    img_piece, *_ = find_point_neighbourhood(intersect.point, self.offset, self.img, line)
                    if not is_court_corner(img_piece):
                        print(intersect.point)
                        continue

                    net_intersection = self._find_closer_outer_netpoint(line, intersect.point, bin_thresh)

                    if net_intersection is not None:
                        return intersect, net_intersection


    def _find_closer_outer_netpoint(self, line: Line, point: Point, bin_thresh: float, warmup: int = 5) -> Intersection | None:
        net_intersection = None
        intersection_global = None
        i = 0
        while net_intersection is None:
            i += 1
            new_point, img_piece, original_range = traverse_line(point, self.offset, self.img, line)

            # sprawdzanie czy linia boczna wewnetrza po drodze
            # jesli tak to break
            if is_inner_sideline(img_piece):
                break

            print(f'{new_point.y=}{point.y=}')
            if new_point.y >= point.y:
                print('point new point')
                break
            else:
                point = new_point

            if i < warmup:
                continue

            net_line_groups = find_net_lines(img_piece, bin_thresh=bin_thresh)

            intersections = []
            local_line = transform_line(line, self.img, *original_range, False)

            # print(net_line_groups)

            # print(local_line.slope, local_line.intercept)
            # print(line.slope, line.intercept)

            # plt.imshow(img_piece)
            # plt.show()

            if check_items_sign(net_line_groups):

                for net_line in net_line_groups:
                    # print(net_line)
                    intersection = local_line.intersection(net_line, img_piece)

                    # print('intersection: ', intersection)

                    # img_copy = img_piece.copy()
                    # pts1 = local_line.limit_to_img(img_copy)
                    # pts2 = net_line.limit_to_img(img_copy)
                    # cv2.line(img_copy, *pts1, (0, 0, 255))
                    # cv2.line(img_copy, *pts2, (0, 255, 255))
                    # plt.imshow(img_copy)
                    # plt.show()

                    # print(net_line, local_line)

                    if intersection is not None and np.sign(net_line.slope) != np.sign(local_line.slope) and net_line.slope != 0: ### SPRAWDZENIE - dodanie nowego warunku tutaj
                        intersections.append(intersection)

            if len(intersections) > 0:
                net_intersection = sorted(intersections, key = lambda intersection: intersection.point.y)[-1]
                intersection_global = transform_intersection(net_intersection, self.img, *original_range)


        return intersection_global, line
    
    
    def find_further_outer_baseline_intersection(self, closer_outer_baseline_intersection: Intersection, used_line: Line, cannys_lower_thresh: int = 20, cannys_upper_thresh: int = 100, warmup: int = 5, offset: int = None) -> Intersection | None:
        further_outer_baseline_intersection = None
        offset = self.netline_offset if not offset else offset
        point = closer_outer_baseline_intersection.point
        unused_line = closer_outer_baseline_intersection.other_line(used_line)
        i = 0
        intersections = []
        while True:
            i += 1
            stop = False
            new_point, img_piece, original_range = traverse_line(point, offset, self.img, unused_line)
            local_line = transform_line(unused_line, self.img, *original_range, False)

            if new_point.y >= point.y:
                print('break point')
                break
            else:
                point = new_point

            if i < warmup:
                continue

            further_outer_baseline_intersection = get_further_outer_baseline_corner(img_piece, local_line, cannys_lower_thresh, cannys_upper_thresh)

            if further_outer_baseline_intersection:
                break
            else:
                continue

        return transform_intersection(further_outer_baseline_intersection, self.img, *original_range), local_line


    def find_netline(self, closer_outer_netpoint: Point, baseline: Line, max_line_gap: int, cannys_thresh_lower: int = 50, cannys_thresh_upper: int = 150, hough_thresh: int = 100, min_line_len: int = 100) -> Line:
        pic_copy = self.img.copy()
        img_gray = cv2.cvtColor(pic_copy, cv2.COLOR_RGB2GRAY)
        inv_img_gray = 255 - img_gray
        edges = cv2.Canny(inv_img_gray, cannys_thresh_lower, cannys_thresh_upper)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap)
        lines = [] if lines is None else lines
            
        line_obj = [Line.from_hough_line(line[0]) for line in lines]
        line_obj = [line for line in line_obj if line.slope is not None and np.sign(line.slope) == np.sign(baseline.slope)]

        return get_closest_line(line_obj, closer_outer_netpoint)
    

    def find_further_doubles_sideline(self, further_outer_baseline_point: Point, prev_local_line: Line, offset: int, extra_offset: int, bin_thresh: float, surface_type: str):
        img_piece, *original_range = find_point_neighbourhood_simple(further_outer_baseline_point, offset + extra_offset, self.img, prev_local_line)

        # plt.imshow(img_piece)
        # plt.show()

        img_gray = cv2.cvtColor(img_piece, cv2.COLOR_RGB2GRAY)
        
        if surface_type == 'clay':
            edges = cv2.Canny(img_gray, 150, 500)

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=50, maxLineGap=10)
            lines = [] if lines is None else lines
            
            line_obj = [Line.from_hough_line(line[0]) for line in lines]
            line_obj = [line for line in line_obj if line.slope is not None and np.sign(line.slope) != np.sign(prev_local_line.slope)]

            
            local_point = transform_point(further_outer_baseline_point, *original_range, False)
            line = get_closest_line(line_obj, local_point)

            # img_copy = img_piece.copy()
            # pts = line.limit_to_img(img_copy)
            # cv2.line(img_copy, *pts, (255, 0, 0))
            # plt.imshow(img_copy)
            # plt.show()

            return transform_line(line, self.img, *original_range)


        bin_img = (img_gray > img_gray.max() * bin_thresh).astype(np.uint8)
        # plt.imshow(bin_img)
        # plt.show()

        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [con for con in contours if len(con) > 0 and (con[:, 0, 1].min() != 0 or con[:, 0, 0].min() != 0)]

        if filtered_contours:
            areas = [cv2.contourArea(con) for con in filtered_contours]
            biggest_contour = filtered_contours[areas.index(max(areas))]

        img_copy = img_piece.copy()
        detected_obj = cv2.drawContours(np.zeros_like(img_copy), contours, areas.index(max(areas)), 255, thickness=1)
        # plt.imshow(detected_obj)
        # plt.show()

        mask = np.zeros_like(bin_img, dtype=np.uint8)
        cv2.fillPoly(mask, [biggest_contour], 1)

        # plt.imshow(mask)
        # plt.show()

        ones = np.argwhere(mask==1)
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

        # img_copy = img_piece.copy()
        # cv2.circle(img_copy, (x1, y1), 1, (255, 0, 0), -1)
        # cv2.circle(img_copy, (x2, y2), 1, (255, 0, 0), -1)

        # plt.imshow(img_copy)
        # plt.show()

        line = Line.from_points(p1, p2)
        return transform_line(line, self.img, *original_range)


    def scan_baseline(self):
        pass