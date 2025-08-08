import numpy as np
from collections import defaultdict
from .lines import Line, Intersection, Point
from .func import traverse_line, find_net_lines, check_items_sign, transform_intersection

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
                    net_intersection = self._find_closer_outer_netpoint(line, intersect.point)

                    if net_intersection is not None:
                        return intersect, net_intersection


    def _find_closer_outer_netpoint(self, line: Line, point: Point) -> Intersection | None:
        net_intersection = None
        intersection_global = None

        while net_intersection is None:
            new_point, img_piece, original_range = traverse_line(point, self.offset, self.img, line)

            if new_point.y > point.y:
                break
            else:
                point = new_point

            net_line_groups = find_net_lines(img_piece)

            intersections = []
            local_line = line.copy()
            local_line.intercept = 0

            # rec = draw_rectangle(self.img, *original_range, self.offset)

            if check_items_sign(net_line_groups):

                for net_line in net_line_groups:
                    intersection = local_line.intersection(net_line, img_piece)
  
                    if intersection is not None:
                        intersections.append(intersection)

            if len(intersections) > 0:
                net_intersection = sorted(intersections, key = lambda intersection: intersection.point.y)[-1]
                intersection_global = transform_intersection(net_intersection, *original_range, img_piece)


        return intersection_global



