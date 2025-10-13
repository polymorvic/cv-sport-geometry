from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class GroundTruthPoint(BaseModel):
    model_config = ConfigDict(strict=True)
    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def within_0_100(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError("Coordinate must be within [0, 100]")
        return v
    

class GroundTruthCourtPoints(BaseModel):
    model_config = ConfigDict(strict=True)
    closer_outer_baseline_point: GroundTruthPoint
    closer_outer_netline_point: GroundTruthPoint
    further_outer_netline_point: GroundTruthPoint
    further_outer_baseline_point: GroundTruthPoint
    closer_inner_baseline_point: GroundTruthPoint
    further_inner_baseline_point: GroundTruthPoint
    closer_inner_netline_point: GroundTruthPoint
    further_inner_netline_point: GroundTruthPoint
    centre_service_point: GroundTruthPoint
    further_service_point: GroundTruthPoint
    closer_service_point: GroundTruthPoint
    net_service_point: GroundTruthPoint


class BinThreshEndlineScan(BaseModel):
    baseline: float
    netline: float


class CannyThreshold(BaseModel):
    lower: int
    upper: int


class ImageParams(BaseModel):
    pic_name: str
    bin_thresh: float
    bin_thresh_endline_scan: BinThreshEndlineScan
    bin_thresh_centre_service_line: float
    max_line_gap: int
    max_line_gap_centre_service_line: int
    hough_thresh: int
    min_line_len_ratio: float
    offset: int
    extra_offset: int
    surface_type: Literal["clay", "another"]
    canny_thresh: CannyThreshold

    class Config:
        strict = True

class TestImageParams(ImageParams):
    ground_truth_points: GroundTruthCourtPoints


class TestConfig(BaseModel):
    testing_pics_dir: str
    run_name: str
    data: list[TestImageParams]


class Settings(BaseSettings):
    debug: bool
    model_config = SettingsConfigDict(env_file=".env")