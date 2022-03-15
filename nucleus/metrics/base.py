import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List

from nucleus.annotation import AnnotationList
from nucleus.prediction import PredictionList


class MetricResult(ABC):
    """Base MetricResult class"""


@dataclass
class ScalarResult(MetricResult):
    """A scalar result contains the value of an evaluation, as well as its weight.
    The weight is useful when aggregating metrics where each dataset item may hold a
    different relative weight. For example, when calculating precision over a dataset,
    the denominator of the precision is the number of annotations, and therefore the weight
    can be set as the number of annotations.

    Attributes:
        value (float): The value of the evaluation result
        weight (float): The weight of the evaluation result.
    """

    value: float
    weight: float = 1.0

    @staticmethod
    def aggregate(results: Iterable["ScalarResult"]) -> "ScalarResult":
        """Aggregates results using a weighted average."""
        results = list(filter(lambda x: x.weight != 0, results))
        total_weight = sum([result.weight for result in results])
        total_value = sum([result.value * result.weight for result in results])
        value = total_value / max(total_weight, sys.float_info.epsilon)
        return ScalarResult(value, total_weight)


class Metric(ABC):
    """Abstract class for defining a metric, which takes a list of annotations
    and predictions and returns a scalar.

    To create a new concrete Metric, override the `__call__` function
    with logic to define a metric between annotations and predictions. ::

        from nucleus import BoxAnnotation, CuboidPrediction, Point3D
        from nucleus.annotation import AnnotationList
        from nucleus.prediction import PredictionList
        from nucleus.metrics import Metric, MetricResult
        from nucleus.metrics.polygon_utils import BoxOrPolygonAnnotation, BoxOrPolygonPrediction

        class MyMetric(Metric):
            def __call__(
                self, annotations: AnnotationList, predictions: PredictionList
            ) -> MetricResult:
                value = (len(annotations) - len(predictions)) ** 2
                weight = len(annotations)
                return MetricResult(value, weight)

        box = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

        cuboid = CuboidPrediction(
            label="car",
            position=Point3D(100, 100, 10),
            dimensions=Point3D(5, 10, 5),
            yaw=0,
            reference_id="pointcloud_1",
            confidence=0.8,
            annotation_id="pointcloud_1_car_cuboid_1",
            metadata={"vehicle_color": "green"}
        )

        metric = MyMetric()
        annotations = AnnotationList(box_annotations=[box])
        predictions = PredictionList(cuboid_predictions=[cuboid])
        metric(annotations, predictions)
    """

    @abstractmethod
    def __call__(
        self, annotations: AnnotationList, predictions: PredictionList
    ) -> MetricResult:
        """A metric must override this method and return a metric result, given annotations and predictions."""

    @abstractmethod
    def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
        """A metric must define how to aggregate results from single items to a single ScalarResult.

        E.g. to calculate a R2 score with sklearn you could define a custom metric class ::

            class R2Result(MetricResult):
                y_true: float
                y_pred: float


        And then define an aggregate_score ::

            def aggregate_score(self, results: List[MetricResult]) -> ScalarResult:
                y_trues = []
                y_preds = []
                for result in results:
                    y_true.append(result.y_true)
                    y_preds.append(result.y_pred)
                r2_score = sklearn.metrics.r2_score(y_trues, y_preds)
                return ScalarResult(r2_score)

        """
