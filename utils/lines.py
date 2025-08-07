from __future__ import annotations
import copy
import numpy as np


class Line:
    """
    Represents a 2D line in either slope-intercept form (y = ax + b) or vertical line form (xv = constant).
    Distinguishes the existance of vertical lines where there is no slope and intercept but constant x-value instead.

    Attributes:
        slope (float | None): The slope (a) of the line. None if the line is vertical.
        intercept (float | None): The y-intercept (b) of the line. None if the line is vertical.
        xv (float | None): The constant x-value for vertical lines. None if the line is not vertical.
    """
    

    def __init__(self, slope: float | None = None, intercept: float | None = None, xv: float | None = None):
        """
        Initializes a Line instance.

        Args:
            slope (float | None, optional): The slope of the line. Defaults to None.
            intercept (float | None, optional): The intercept of the line. Defaults to None.
            xv (float | None, optional): The constant x-value for vertical lines. Defaults to None.
        """
        self.slope = slope
        self.intercept = intercept
        self.xv = xv


    def __key(self):
        """
        Returns a tuple of identifying attributes used for hashing and equality comparison.

        Returns:
            tuple: A tuple containing slope, intercept, and xv.
        """
        return (self.slope, self.intercept, self.xv)

    
    def __hash__(self):
        """
        Returns the hash of the line based on its identifying attributes.

        Returns:
            int: Hash value of the line.
        """
        return hash(self.__key())


    def __eq__(self, other):
        """
        Checks if this line is equal to another line.

        Args:
            other (Line): Another Line object to compare with.

        Returns:
            bool: True if both lines are equal, False otherwise.
        """
        if isinstance(other, Line):
            return self.__key() == other.__key()
        return NotImplemented


    def __repr__(self):
        """
        Returns a string representation of the line.

        Returns:
            str: String representation of the line.
        """
        return f'y = {self.slope} * x + {self.slope}'
    

    def copy(self):
        """
        Creates a deep copy of the line.

        Returns:
            Line: A new Line instance with the same attributes.
        """
        return copy.deepcopy(self)
    

    def intersection(self, another_line: Line, image: np.ndarray):
        raise NotImplementedError
    

    def transform_to_another_coordinate_system(self, source_img: np.ndarray, dst_image: np.ndarray, offset: int) -> Line:
        raise NotImplementedError
        

    def y_for_x(self, x: int) -> int | None:
        """
        Calculates the y-coordinate on the line for a given x-coordinate.
        It handles when line instance is vertical, then return None because y doesnt exist.

        Args:
            x (int): The x-coordinate.

        Returns:
            int | None: The corresponding y-coordinate if the line is not vertical, otherwise None.

        Note:
            The return value must be an integer because we are working with images,
            and pixel coordinates must be whole numbers.
        """
        if self.slope is None or self.intercept is None:
            return None
        return int(self.slope * x + self.intercept)
    

    def x_for_y(self, y):
        """
        Calculates the x-coordinate on the line for a given y-coordinate.
        It handles when line instance is vertical or horizontal.
        If Vertical line: x is constant, in case of horizontal line or undefined slope: no unique x for given y

        Args:
            y (int): The y-coordinate.

        Returns:
            int | None: The corresponding x-coordinate as an integer if the line is not horizontal;
                        or the stored x-value if the line is vertical; otherwise, None.

        Note:
            The return value must be an integer because we are working with images,
            and pixel coordinates must be whole numbers.
        """
        if self.xv is not None:
            return int(self.xv)
        if self.slope == 0 or self.slope is None:
            return None
        return int((y - self.intercept) / self.slope)
    

    def get_points_by_distance(self, main_point: tuple[int, int], distance: float) -> tuple[int, int]:
        """
        Finds two points on the line that are at a given Euclidean distance from a specified point.

        Args:
            main_point (tuple[int, int]): The reference (x, y) point from which distance is measured.
            distance (float): The Euclidean distance to measure along the line.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: Two (x, y) integer coordinate points on the line.

        Note:
            Pixel coordinates are returned as integers.
            For vertical lines, points are offset along the y-axis.
        """
        main_x, main_y = main_point

        # Vertical line: return two points offset by ±distance along the y-axis
        if self.xv is not None:
            return (int(main_x), int(main_y - distance)), (int(main_x), int(main_y + distance))

        if self.slope is None or self.intercept is None:
            raise ValueError("Cannot compute points: line is not properly defined.")

        m = self.slope
        b = self.intercept

        # Solve: (x - main_x)^2 + (a*x + b - main_y)^2 = distance^2
        # This simplifies to a quadratic equation in x: A*x^2 + B*x + C = 0
        A = 1 + m**2
        B = -2 * main_x + 2 * m * (b - main_y)
        C = main_x**2 + (b - main_y)**2 - distance**2

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            # No real intersection: distance is too large or point is far from the line
            raise ValueError("No real solution: check if the distance is too large or the point is far from the line.")

        sqrt_delta = np.sqrt(discriminant)

        x1 = (-B + sqrt_delta) / (2 * A)
        x2 = (-B - sqrt_delta) / (2 * A)

        # Get corresponding y values using the line equation
        y1 = self.y_for_x(x1)
        y2 = self.y_for_x(x2)

        return (int(x1), int(y1)), (int(x2), int(y2))
        
    
    def limit_to_img(self, img: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Returns two endpoints of the line segment clipped to the image boundaries.

        Args:
            img (np.ndarray): The image array used to determine dimensions.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: Two (x, y) points that define the visible
            part of the line within the image.
        """
        img_width, img_height = img.shape[1] - 1, img.shape[0] - 1

        if self.xv is not None:
            x = int(self.xv)
            return (x, 0), (x, img_height)

        if self.slope == 0:
            y = int(self.intercept)
            return (0, y), (img_width, y)

        points = []

        x_top = self.x_for_y(0)
        if x_top is not None and 0 <= x_top <= img_width:
            points.append((int(x_top), 0))

        x_bottom = self.x_for_y(img_height)
        if x_bottom is not None and 0 <= x_bottom <= img_width:
            points.append((int(x_bottom), img_height))

        y_left = self.y_for_x(0)
        if y_left is not None and 0 <= y_left <= img_height:
            points.append((0, int(y_left)))

        y_right = self.y_for_x(img_width)
        if y_right is not None and 0 <= y_right <= img_height:
            points.append((img_width, int(y_right)))

        unique_points = list(dict.fromkeys(points))  # removes duplicates while preserving order

        if len(unique_points) >= 2:
            return unique_points[0], unique_points[1]

        raise ValueError("Line does not intersect the image in at least two places.")


    def check_point_on_line(self, point: tuple[int, int], tolerance: int = None) -> bool:
        """
        Checks whether a given point lies on the line, optionally within a specified tolerance.

        Args:
            point (tuple[int, int]): The (x, y) coordinates of the point to check.
            tolerance (int, optional): Allowed deviation in pixels for both x and y. 
                                    If None, the match must be exact.

        Returns:
            bool: True if the point lies on the line (within tolerance if provided), False otherwise.
        """
        point_x, point_y = point

        y = self.y_for_x(point_x)
        x = self.x_for_y(point_y)

        # If point is outside defined part of the line
        if y is None or x is None:
            return False

        line_point_y = int(y)
        line_point_x = int(x)

        if tolerance is None:
            return point_x == line_point_x and point_y == line_point_y

        return (
            abs(line_point_y - point_y) < tolerance and
            abs(line_point_x - point_x) < tolerance
        )  


    @property
    def theta(self) -> float:
        """
        Returns the angle (in degrees) between the line and the horizontal axis.

        Returns:
            float: The angle in degrees. For vertical lines, returns 90.
        """
        if self.slope is None:
            return 90.0
        return np.degrees(np.arctan(self.slope))
        

    @classmethod
    def from_hough_line(cls, hough_line: tuple[int, int, int, int]):
        """
        Creates a Line instance from a Hough line segment represented by two points.

        Args:
            hough_line (tuple[int, int, int, int]): A 4-tuple (x1, y1, x2, y2) representing the endpoints of the line.

        Returns:
            Line: A Line object representing the line segment.
        """
        x1, y1, x2, y2 = hough_line
        return cls.from_points((x1, y1), (x2, y2))
    

    @classmethod
    def from_points(cls, p1: tuple[int, int], p2: tuple[int, int]) -> Line:
        """
        Creates a Line instance from two points.

        Args:
            p1 (tuple[int, int]): The first point (x1, y1).
            p2 (tuple[int, int]): The second point (x2, y2).

        Returns:
            Line: A Line object defined by the two points.
        """
        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2:
            # Vertical line
            slope, intercept = None, None
            xv = x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            xv = None

        return cls(slope, intercept, xv)
    


    

