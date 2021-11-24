import abc
from typing import List

from nucleus.modelci import EvalFunctionResponse


class EvalFunction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def from_response(self):
        pass


class IoU:
    def __init__(self):
        pass


class MeanAveragePrecisionForBoxes:
    pass


class AvailableEvaluationFunctions:
    """Collection class that acts as a common entrypoint to access evaluation function.

    e = client.model_ci.eval_functions

    """

    def __init__(self, available_functions: List[EvalFunctionResponse]):
        """Class that holds all evaluation functions """
        # Names taken from: https://github.com/scaleapi/models/pull/2100/files
        name_to_func = {f.name: f for f in available_functions}
        self.iou = (
            IoU.from_response(name_to_func["iou"])
            if "iou" in name_to_func
            else None
        )
        self.mean_average_precision_for_boxes = (
            MeanAveragePrecisionForBoxes.from_response(
                name_to_func["mean_average_precision_boxes"]
            )
            if "mean_average_precision_for_boxes" in name_to_func
            else None
        )
