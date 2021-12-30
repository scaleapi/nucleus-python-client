import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Union

from nucleus.annotation import Annotation
from nucleus.prediction import Prediction

AnnotationOrPredictionList = Union[List[Annotation], List[Prediction]]


@dataclass
class MetricResult:
    """An Metric Result contains the value of an evaluation, as well as its weight.
    The weight is useful when aggregating metrics where each dataset item may hold a
    different relative weight. For example, when calculating precision over a dataset,
    the denominator of the precision is the number of annotations, and therefore the weight
    can be set as the number of annotations.

    Attributes:
        value (float): The value of the evaluation result
        weight (float): The weight of the evaluation result.
    """

    value: float
    weight: float = 1

    @classmethod
    def aggregate(cls, results: Iterable["MetricResult"]) -> "MetricResult":
        """Aggregates results using a weighted average."""
        results = list(filter(lambda x: x.weight != 0, results))
        total_weight = sum([result.weight for result in results])
        total_value = sum([result.value * result.weight for result in results])
        value = total_value / max(total_weight, sys.float_info.epsilon)
        return cls(value, total_weight)


class Metric(ABC):
    """Abstract class to define a Metric."""

    @abstractmethod
    def __call__(
        self, annotations: List[Annotation], predictions: List[Prediction]
    ) -> MetricResult:
        """A metric must override this method and return a metric result, given annotations and predictions."""


class Filter(ABC):
    """Abstract class to define an Filter"""

    @abstractmethod
    def __call__(
        self, annotations: AnnotationOrPredictionList
    ) -> AnnotationOrPredictionList:
        """A Filter must override this method."""


class AnnotationFilter(ABC):
    """Abstract class to define an Filter"""

    @abstractmethod
    def __call__(self, annotations: List[Annotation]) -> List[Annotation]:
        """An Annotation Filter must override this method."""


class PredictionFilter(ABC):
    """Abstract class to define an Prediction Filter"""

    @abstractmethod
    def __call__(self, predictions: List[Prediction]) -> List[Prediction]:
        """An Prediction Filter must override this method."""
