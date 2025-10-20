from datetime import datetime

from .schemas import Settings

ARRAY_X_INDEX = 1

ARRAY_Y_INDEX = 0

WIDTH = int(10.973 * 1000)

LENGTH = int(23.77 * 1000)

DIST_OUTER_SIDELINE = int(1.372 * 1000)

COURT_LENGTH_HALF = int(LENGTH / 2)

COURT_WIDTH_HALF = int(WIDTH / 2)

DIST_FROM_BASELINE = int(5.485 * 1000)

SETTINGS = Settings()

TIMESTAMP_STRING = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

SCENARIOS = [
    (
        "test1",
        [
            "closer_outer_baseline_point",
            "closer_outer_netline_point",
            "closer_inner_netline_point",
            "closer_inner_baseline_point",
        ],
    ),
    (
        "test2",
        [
            "closer_outer_baseline_point",
            "closer_inner_baseline_point",
            "further_inner_baseline_point",
            "further_outer_baseline_point",
            "further_outer_netline_point",
            "net_service_point",
            "closer_inner_netline_point",
            "closer_outer_netline_point",
        ],
    ),
    (
        "test3",
        [
            "closer_outer_baseline_point",
            "closer_inner_baseline_point",
            "closer_service_point",
            "centre_service_point",
            "further_service_point",
        ],
    ),
    (
        "test4",
        [
            "further_outer_baseline_point",
            "further_inner_baseline_point",
            "further_service_point",
            "centre_service_point",
            "net_service_point",
        ],
    ),
]
