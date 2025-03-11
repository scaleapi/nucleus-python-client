import itertools
from typing import Callable, Dict, List, Optional, Union

from nucleus.validate.eval_functions.base_eval_function import (
    EvalFunctionConfig,
)

from ...metrics.filtering import ListOfAndFilters, ListOfOrAndFilters
from ..data_transfer_objects.eval_function import EvalFunctionEntry
from ..errors import EvalFunctionNotAvailableError
from .config_classes.segmentation import (
    SegmentationFWAVACCConfig,
    SegmentationIOUConfig,
    SegmentationMAPConfig,
    SegmentationPrecisionConfig,
    SegmentationRecallConfig,
)

# TODO(gunnar) split up into modules
# pylint: disable=too-many-lines


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
            poly_iou: BoundingBoxIOU = client.validate.eval_functions.poly_iou
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_iou(confidence_threshold=0.8) > 0.5]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "poly_iou"


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
            poly_map: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.poly_map
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_map(iou_threshold=0.6) > 0.8]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "poly_map"


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
            poly_recall: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.poly_recall
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_recall(iou_threshold=0.6, confidence_threshold=0.4) > 0.9]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "poly_recall"


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
            poly_precision: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.poly_precision
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_precision(iou_threshold=0.6, confidence_threshold=0.4) > 0.9]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "poly_precision"


class SegmentationToPolyIOUConfig(EvalFunctionConfig):
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
            poly_iou: BoundingBoxIOU = client.validate.eval_functions.poly_iou
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_iou(confidence_threshold=0.8) > 0.5]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "segmentation_to_poly_iou"


class SegmentationToPolyMAPConfig(EvalFunctionConfig):
    def __call__(
        self,
        iou_thresholds: Union[List[float], str] = "coco",
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
            poly_map: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.poly_map
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_map(iou_threshold=0.6) > 0.8]
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
            iou_thresholds=iou_thresholds,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "segmentation_to_poly_map"


class SegmentationToPolyRecallConfig(EvalFunctionConfig):
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
            poly_recall: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.poly_recall
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_recall(iou_threshold=0.6, confidence_threshold=0.4) > 0.9]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "segmentation_to_poly_recall"


class SegmentationToPolyPrecisionConfig(EvalFunctionConfig):
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
            poly_precision: BoundingBoxMeanAveragePrecision= client.validate.eval_functions.poly_precision
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_precision(iou_threshold=0.6, confidence_threshold=0.4) > 0.9]
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "segmentation_to_poly_precision"


class SegmentationToPolyAveragePrecisionConfig(EvalFunctionConfig):
    def __call__(
        self,
        label: str = "label",
        iou_threshold: float = 0.5,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Initializes SegmentationToPolyAveragePrecision object.

        Args:
            iou_threshold: IOU threshold to consider detection as valid. Must be in [0, 1]. Default 0.5
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
            label=label,
            iou_threshold=iou_threshold,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "segmentation_to_poly_ap"


class BoundingBoxIOUConfig(EvalFunctionConfig):
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
        """Configures a call to :class:`BoundingBoxIOU` object.
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_iou"


class BoundingBoxMAPConfig(EvalFunctionConfig):
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
        """Configures a call to :class:`BoundingBoxMAP` object.
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_map"


class BoundingBoxRecallConfig(EvalFunctionConfig):
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
        """Configures a call to :class:`BoundingBoxRecall` object.
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "bbox_recall"


class BoundingBoxPrecisionConfig(EvalFunctionConfig):
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
        """Configures a call to :class:`BoundingBoxPrecision` object.
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
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
            confidence_threshold=confidence_threshold,
            f1_method=f1_method,
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "cat_f1"


class CustomEvalFunction(EvalFunctionConfig):
    @classmethod
    def expected_name(cls) -> str:
        raise NotImplementedError(
            "Custom evaluation functions are coming soon"
        )  # Placeholder: See super().eval_func_entry for actual name


class ExternalEvalFunction(EvalFunctionConfig):
    def __call__(self, **kwargs):
        raise NotImplementedError("Cannot call an external function")

    @classmethod
    def expected_name(cls) -> str:
        return "external_function"


class StandardEvalFunction(EvalFunctionConfig):
    """Class for standard Model CI eval functions that have not been added as attributes on
    AvailableEvalFunctions yet.
    """

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
    BoundingBoxIOUConfig,
    BoundingBoxMAPConfig,
    BoundingBoxPrecisionConfig,
    BoundingBoxRecallConfig,
    CuboidRecallConfig,
    CuboidIOU2DConfig,
    CuboidIOU3DConfig,
    CuboidPrecisionConfig,
    CategorizationF1Config,
    CustomEvalFunction,
    ExternalEvalFunction,
    EvalFunctionNotAvailable,
    StandardEvalFunction,
    PolygonMAPConfig,
    PolygonIOUConfig,
    PolygonRecallConfig,
    PolygonPrecisionConfig,
    SegmentationToPolyRecallConfig,
    SegmentationToPolyIOUConfig,
    SegmentationToPolyMAPConfig,
    SegmentationToPolyPrecisionConfig,
    SegmentationToPolyAveragePrecisionConfig,
    SegmentationFWAVACCConfig,
    SegmentationIOUConfig,
    SegmentationPrecisionConfig,
    SegmentationRecallConfig,
    SegmentationMAPConfig,
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
            if not f.is_public and not f.is_external_function
        }
        self._external_to_function: Dict[str, ExternalEvalFunction] = {
            f.name: ExternalEvalFunction(f)
            for f in available_functions
            if f.is_external_function
        }
        self.bbox_iou: BoundingBoxIOUConfig = (
            self._assign_eval_function_if_defined(BoundingBoxIOUConfig)
        )  # type: ignore
        self.bbox_precision: BoundingBoxPrecisionConfig = self._assign_eval_function_if_defined(
            BoundingBoxPrecisionConfig  # type: ignore
        )
        self.bbox_recall: BoundingBoxRecallConfig = self._assign_eval_function_if_defined(
            BoundingBoxRecallConfig  # type: ignore
        )
        self.bbox_map: BoundingBoxMAPConfig = self._assign_eval_function_if_defined(
            BoundingBoxMAPConfig  # type: ignore
        )
        self.cat_f1: CategorizationF1Config = self._assign_eval_function_if_defined(
            CategorizationF1Config  # type: ignore
        )
        self.cuboid_iou_2d: CuboidIOU2DConfig = self._assign_eval_function_if_defined(CuboidIOU2DConfig)  # type: ignore
        self.cuboid_iou_3d: CuboidIOU3DConfig = self._assign_eval_function_if_defined(CuboidIOU3DConfig)  # type: ignore
        self.cuboid_precision: CuboidPrecisionConfig = (
            self._assign_eval_function_if_defined(CuboidPrecisionConfig)
        )  # type: ignore
        self.cuboid_recall: CuboidRecallConfig = (
            self._assign_eval_function_if_defined(CuboidRecallConfig)
        )  # type: ignore
        self.poly_iou: PolygonIOUConfig = self._assign_eval_function_if_defined(PolygonIOUConfig)  # type: ignore
        self.poly_precision: PolygonPrecisionConfig = self._assign_eval_function_if_defined(
            PolygonPrecisionConfig  # type: ignore
        )
        self.poly_recall: PolygonRecallConfig = self._assign_eval_function_if_defined(
            PolygonRecallConfig  # type: ignore
        )
        self.poly_map: PolygonMAPConfig = self._assign_eval_function_if_defined(
            PolygonMAPConfig  # type: ignore
        )
        self.segmentation_to_poly_iou: SegmentationToPolyIOUConfig = (
            self._assign_eval_function_if_defined(SegmentationToPolyIOUConfig)
        )  # type: ignore
        self.segmentation_to_poly_precision: SegmentationToPolyPrecisionConfig = self._assign_eval_function_if_defined(
            SegmentationToPolyPrecisionConfig  # type: ignore
        )
        self.segmentation_to_poly_recall: SegmentationToPolyRecallConfig = self._assign_eval_function_if_defined(
            SegmentationToPolyRecallConfig  # type: ignore
        )
        self.segmentation_to_poly_map: SegmentationToPolyMAPConfig = self._assign_eval_function_if_defined(
            SegmentationToPolyMAPConfig  # type: ignore
        )
        self.segmentation_to_poly_ap: SegmentationToPolyAveragePrecisionConfig = self._assign_eval_function_if_defined(
            SegmentationToPolyAveragePrecisionConfig  # type: ignore
        )

        self.seg_iou: SegmentationIOUConfig = self._assign_eval_function_if_defined(
            SegmentationIOUConfig  # type: ignore
        )
        self.seg_recall: SegmentationRecallConfig = self._assign_eval_function_if_defined(
            SegmentationRecallConfig  # type: ignore
        )
        self.seg_map: SegmentationMAPConfig = self._assign_eval_function_if_defined(
            SegmentationMAPConfig  # type: ignore
        )
        self.seg_precision: SegmentationPrecisionConfig = self._assign_eval_function_if_defined(
            SegmentationPrecisionConfig  # type: ignore
        )
        self.seg_fwavacc: SegmentationFWAVACCConfig = self._assign_eval_function_if_defined(
            SegmentationFWAVACCConfig  # type: ignore
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
            f"<AvailableEvaluationFunctions: public: {functions_lower} "
            f"private: {list(self._custom_to_function.keys())} "
            f"external: {list(self._external_to_function.keys())}"
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
        """Private functions uploaded to Model CI

        Returns:
            Dict of function name to :class:`CustomEvalFunction`.
        """
        return self._custom_to_function

    @property
    def external_functions(self) -> Dict[str, ExternalEvalFunction]:
        """External functions uploaded to Model CI

        Returns:
            Dict of function name to :class:`ExternalEvalFunction`.
        """
        return self._external_to_function

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
            self._external_to_function.values(),
        ):
            if eval_func.id == eval_function_id:
                return eval_func
        raise EvalFunctionNotAvailableError(
            f"Could not find Eval Function with id {eval_function_id}"
        )
