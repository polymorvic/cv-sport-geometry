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

    
    def find_closer_outer_baseline_point(self) -> tuple[Intersection, Point]:
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

                    net_intersection = self._find_closer_outer_netpoint(line, intersect.point)

                    if net_intersection is not None:
                        return intersect, net_intersection


    def _find_closer_outer_netpoint(self, line: Line, point: Point, warmup: int = 5) -> Intersection | None:
        net_intersection = None
        intersection_global = None
        i = 0
        while net_intersection is None:
            i += 1
            new_point, img_piece, original_range = traverse_line(point, self.offset, self.img, line)

            # sprawdzanie czy linia boczna wewnetrza po drodze
            # jesli tak to break
            # if is_inner_sideline(img_piece):
            #     break

            print(f'{new_point.y=}{point.y=}')
            if new_point.y >= point.y:
                print('point new point')
                break
            else:
                point = new_point

            if i < warmup:
                continue

            net_line_groups = find_net_lines(img_piece, bin_thresh=0.8)

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


        return intersection_global



