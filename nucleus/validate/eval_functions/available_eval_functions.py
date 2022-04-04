import itertools
from typing import Callable, Dict, List, Optional, Union

from nucleus.logger import logger
from nucleus.validate.eval_functions.base_eval_function import (
    EvalFunctionConfig,
)

from ...metrics.filtering import ListOfAndFilters, ListOfOrAndFilters
from ..data_transfer_objects.eval_function import EvalFunctionEntry
from ..errors import EvalFunctionNotAvailableError


class PolygonIOUConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`PolygonIOU` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            bbox_iou: BoundingBoxIOU = client.validate.eval_functions.bbox_iou
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[bbox_iou(confidence_threshold=0.8) > 0.5]
            )

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_iou"


class PolygonMAPConfig(EvalFunctionConfig):
    def __call__(
        self,
        iou_threshold: float = 0.5,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`PolygonMAP` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            bbox_map: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.bbox_map
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[bbox_map(iou_threshold=0.6) > 0.8]
            )

        Args:
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            iou_threshold=iou_threshold,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_map"


class PolygonRecallConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`PolygonRecall` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            bbox_recall: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.bbox_recall
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[bbox_recall(iou_threshold=0.6, confidence_threshold=0.4) > 0.9]
            )

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_recall"


class PolygonPrecisionConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = False,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`PolygonPrecision` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            bbox_precision: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.bbox_precision
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[bbox_precision(iou_threshold=0.6, confidence_threshold=0.4) > 0.9]
            )

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to False
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_precision"


class CuboidIOU2DConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configure a call to CuboidIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            iou_2d=True,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "cuboid_iou_2d"


class CuboidIOU3DConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configure a call to CuboidIOU object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            iou_2d: whether to return the BEV 2D IOU if true, or the 3D IOU if false.
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            iou_2d=False,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "cuboid_iou_3d"


class CuboidPrecisionConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configure a call to CuboidPrecision object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            iou_2d: whether to return the BEV 2D IOU if true, or the 3D IOU if false.
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "cuboid_precision"


class CuboidRecallConfig(EvalFunctionConfig):
    def __call__(
        self,
        enforce_label_match: bool = True,
        iou_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configure a call to a CuboidRecall object.

        Args:
            enforce_label_match: whether to enforce that annotation and prediction labels must match. Defaults to True
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.0
            iou_2d: whether to return the BEV 2D IOU if true, or the 3D IOU if false.
            confidence_threshold: minimum confidence threshold for predictions. Must be in [0, 1]. Default 0.0
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            enforce_label_match=enforce_label_match,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "cuboid_recall"


class CategorizationF1Config(EvalFunctionConfig):
    def __call__(
        self,
        confidence_threshold: Optional[float] = None,
        f1_method: Optional[str] = None,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """ Configure an evaluation of :class:`CategorizationF1`.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            cat_f1: CategorizationF1 = client.validate.eval_functions.cat_f1
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[cat_f1(confidence_threshold=0.6, f1_method="weighted") > 0.7]
            )

        Args:
            confidence_threshold: minimum confidence threshold for predictions to be taken into account for evaluation.
                 Must be in [0, 1]. Default 0.0
            f1_method: {'micro', 'macro', 'samples','weighted', 'binary'}, \
                default='macro'
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
            annotation_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
            prediction_filters: Filter predicates. Allowed formats are:
                ListOfAndFilters where each Filter forms a chain of AND predicates.
                    or
                ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
                [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
                DNF allows arbitrary boolean logical combinations of single field predicates. The innermost structures
                each describe a single column predicate. The list of inner predicates is interpreted as a conjunction
                (AND), forming a more selective `and` multiple field predicate.
                Finally, the most outer list combines these filters as a disjunction (OR).
        """
        return super().__call__(
            confidence_threshold=confidence_threshold, f1_method=f1_method
        )

    @classmethod
    def expected_name(cls) -> str:
        return "cat_f1"


class CustomEvalFunction(EvalFunctionConfig):
    @classmethod
    def expected_name(cls) -> str:
        raise NotImplementedError(
            "Custm evaluation functions are coming soon"
        )  # Placeholder: See super().eval_func_entry for actual name


class StandardEvalFunction(EvalFunctionConfig):
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


class EvalFunctionNotAvailable(EvalFunctionConfig):
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
    PolygonIOUConfig,
    PolygonMAPConfig,
    PolygonPrecisionConfig,
    PolygonRecallConfig,
    CuboidRecallConfig,
    CuboidIOU2DConfig,
    CuboidIOU3DConfig,
    CuboidPrecisionConfig,
    CategorizationF1Config,
    CustomEvalFunction,
    EvalFunctionNotAvailable,
    StandardEvalFunction,
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
        self._public_to_function: Dict[str, EvalFunctionConfig] = {}
        self._custom_to_function: Dict[str, CustomEvalFunction] = {
            f.name: CustomEvalFunction(f)
            for f in available_functions
            if not f.is_public
        }
        self.bbox_iou: PolygonIOUConfig = self._assign_eval_function_if_defined(PolygonIOUConfig)  # type: ignore
        self.bbox_precision: PolygonPrecisionConfig = self._assign_eval_function_if_defined(
            PolygonPrecisionConfig  # type: ignore
        )
        self.bbox_recall: PolygonRecallConfig = self._assign_eval_function_if_defined(
            PolygonRecallConfig  # type: ignore
        )
        self.bbox_map: PolygonMAPConfig = self._assign_eval_function_if_defined(
            PolygonMAPConfig  # type: ignore
        )
        self.cat_f1: CategorizationF1Config = self._assign_eval_function_if_defined(
            CategorizationF1Config  # type: ignore
        )
        self.cuboid_iou_2d: CuboidIOU2DConfig = self._assign_eval_function_if_defined(CuboidIOU2DConfig)  # type: ignore
        self.cuboid_iou_3d: CuboidIOU3DConfig = self._assign_eval_function_if_defined(CuboidIOU3DConfig)  # type: ignore
        self.cuboid_precision: CuboidPrecisionConfig = self._assign_eval_function_if_defined(CuboidPrecisionConfig)  # type: ignore
        self.cuboid_recall: CuboidRecallConfig = self._assign_eval_function_if_defined(CuboidRecallConfig)  # type: ignore

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
    def public_functions(self) -> Dict[str, EvalFunctionConfig]:
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
