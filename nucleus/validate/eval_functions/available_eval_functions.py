import itertools
from typing import Callable, Dict, List, Type, Union

from nucleus.logger import logger
from nucleus.validate.eval_functions.base_eval_function import BaseEvalFunction

from ..data_transfer_objects.eval_function import EvalFunctionEntry
from ..errors import EvalFunctionNotAvailableError

MEAN_AVG_PRECISION_NAME = "mean_average_precision_boxes"


class BoundingBoxIOU(BaseEvalFunction):
    @classmethod
    def expected_name(cls) -> str:
        return "bbox_iou"


class BoundingBoxMeanAveragePrecision(BaseEvalFunction):
    @classmethod
    def expected_name(cls) -> str:
        return "bbox_map"


class BoundingBoxRecall(BaseEvalFunction):
    @classmethod
    def expected_name(cls) -> str:
        return "bbox_recall"


class BoundingBoxPrecision(BaseEvalFunction):
    @classmethod
    def expected_name(cls) -> str:
        return "bbox_precision"


class CategorizationF1(BaseEvalFunction):
    @classmethod
    def expected_name(cls) -> str:
        return "cat_f1"


class CustomEvalFunction(BaseEvalFunction):
    @classmethod
    def expected_name(cls) -> str:
        raise NotImplementedError(
            "Custm evaluation functions are coming soon"
        )  # Placeholder: See super().eval_func_entry for actual name


class StandardEvalFunction(BaseEvalFunction):
    """Class for standard Model CI eval functions that have not been added as attributes on
    AvailableEvalFunctions yet.
    """

    def __init__(self, eval_function_entry: EvalFunctionEntry):
        logger.warning(
            "Standard function %s not implemented as an attribute on AvailableEvalFunctions",
            eval_function_entry.name,
        )
        super().__init__(eval_function_entry)

    @classmethod
    def expected_name(cls) -> str:
        return "public_function"  # Placeholder: See super().eval_func_entry for actual name


class EvalFunctionNotAvailable(BaseEvalFunction):
    def __init__(
        self, not_available_name: str
    ):  # pylint: disable=super-init-not-called
        self.not_available_name = not_available_name

    def __call__(self, *args, **kwargs):
        self._raise_error()

    def _op_to_test_metric(self, *args, **kwargs):
        self._raise_error()

    def _raise_error(self):
        raise EvalFunctionNotAvailableError(
            f"Eval function '{self.not_available_name}' is not available to the current user. "
            f"Is Model CI enabled for the user?"
        )

    @classmethod
    def expected_name(cls) -> str:
        return "public_function"  # Placeholder: See super().eval_func_entry for actual name


EvalFunction = Union[
    Type[BoundingBoxIOU],
    Type[BoundingBoxMeanAveragePrecision],
    Type[BoundingBoxPrecision],
    Type[BoundingBoxRecall],
    Type[CustomEvalFunction],
    Type[EvalFunctionNotAvailable],
    Type[StandardEvalFunction],
]


class AvailableEvalFunctions:
    """Collection class that acts as a common entrypoint to access evaluation functions. Standard evaluation functions
    provided by Scale are attributes of this class.

    The available evaluation functions are listed in the sample below::

        e = client.validate.eval_functions
        unit_test_criteria = [
            e.bbox_iou() > 5,
            e.bbox_map() > 0.95,
            e.bbox_precision() > 0.8,
            e.bbox_recall() > 0.5,
        ]
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, available_functions: List[EvalFunctionEntry]):
        assert (
            available_functions
        ), "Passed no available functions for current user. Is the feature flag enabled?"
        self._public_func_entries: Dict[str, EvalFunctionEntry] = {
            f.name: f for f in available_functions if f.is_public
        }
        # NOTE: Public are assigned
        self._public_to_function: Dict[str, BaseEvalFunction] = {}
        self._custom_to_function: Dict[str, CustomEvalFunction] = {
            f.name: CustomEvalFunction(f)
            for f in available_functions
            if not f.is_public
        }
        self.bbox_iou = self._assign_eval_function_if_defined(BoundingBoxIOU)  # type: ignore
        self.bbox_precision = self._assign_eval_function_if_defined(
            BoundingBoxPrecision  # type: ignore
        )
        self.bbox_recall = self._assign_eval_function_if_defined(
            BoundingBoxRecall  # type: ignore
        )
        self.bbox_map = self._assign_eval_function_if_defined(
            BoundingBoxMeanAveragePrecision  # type: ignore
        )
        self.cat_f1 = self._assign_eval_function_if_defined(
            CategorizationF1  # type: ignore
        )

        # Add public entries that have not been implemented as an attribute on this class
        for func_entry in self._public_func_entries.values():
            if func_entry.name not in self._public_to_function:
                self._public_to_function[
                    func_entry.name
                ] = StandardEvalFunction(func_entry)

    def __repr__(self):
        """Standard functions are ones Scale provides and custom ones customer defined"""
        # NOTE: setting to lower to be consistent with attribute names
        functions_lower = [
            str(name).lower() for name in self._public_func_entries.keys()
        ]
        return (
            f"<AvailableEvaluationFunctions: public:{functions_lower} "
            f"private: {list(self._custom_to_function.keys())}"
        )

    @property
    def public_functions(self) -> Dict[str, BaseEvalFunction]:
        """Standard functions provided by Model CI.

        Notes:
            These functions are also available as attributes on :class:`AvailableEvalFunctions`

        Returns:
            Dict of function name to :class:`BaseEvalFunction`.
        """
        return self._public_to_function

    @property
    def private_functions(self) -> Dict[str, CustomEvalFunction]:
        """Custom functions uploaded to Model CI

        Returns:
            Dict of function name to :class:`CustomEvalFunction`.
        """
        return self._custom_to_function

    def _assign_eval_function_if_defined(
        self,
        eval_function_constructor: Callable[[EvalFunctionEntry], EvalFunction],
    ):
        """Helper function for book-keeping and assignment of standard Scale provided functions that are accessible
        via attribute access
        """
        # TODO(gunnar): Too convoluted .. simplify
        expected_name = eval_function_constructor.expected_name()  # type: ignore
        if expected_name in self._public_func_entries:
            definition = self._public_func_entries[expected_name]
            eval_function = eval_function_constructor(definition)
            self._public_to_function[expected_name] = eval_function  # type: ignore
            return eval_function
        else:
            return EvalFunctionNotAvailable(expected_name)

    def from_id(self, eval_function_id: str):
        for eval_func in itertools.chain(
            self._public_to_function.values(),
            self._custom_to_function.values(),
        ):
            if eval_func.id == eval_function_id:
                return eval_func
        raise EvalFunctionNotAvailableError(
            f"Could not find Eval Function with id {eval_function_id}"
        )
