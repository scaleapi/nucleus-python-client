from enum import Enum
from dataclasses import dataclass


@dataclass
class ThresholdComparison(str, Enum):
    GREATER_THAN = ("greater_than",)
    GREATER_THAN_EQUAL_TO = ("greater_than_equal_to",)
    LESS_THAN = ("less_than",)
    LESS_THAN_EQUAL_TO = ("less_than_equal_to",)
