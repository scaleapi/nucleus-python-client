import copy
import enum
import functools
import logging
from enum import Enum
from typing import (
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from rich.console import Console
from rich.table import Table

from nucleus.annotation import (
    AnnotationList,
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    LineAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    Segment,
    SegmentationAnnotation,
)
from nucleus.prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    LinePrediction,
    PolygonPrediction,
    PredictionList,
    SegmentationPrediction,
)


class FilterOp(str, Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "="
    EQEQ = "=="
    NEQ = "!="
    IN = "in"
    NOT_IN = "not in"


class FilterType(str, enum.Enum):
    """The type of the filter decides the getter used for the comparison.
    Attributes:
        FIELD: Access the attribute field of an object
        METADATA: Access the metadata dictionary of an object
        SEGMENT_FIELD: Filter segments of a segmentation mask to be considered on segment fields
        SEGMENT_METADATA: Filter segments of a segmentation mask based on segment metadata
    """

    FIELD = "field"
    METADATA = "metadata"
    SEGMENT_FIELD = "segment_field"
    SEGMENT_METADATA = "segment_metadata"


FilterableBaseVals = Union[str, float, int, bool]
FilterableTypes = Union[
    FilterableBaseVals,
    Sequence[FilterableBaseVals],
    Set[FilterableBaseVals],
    Iterable[FilterableBaseVals],
]

AnnotationTypes = Union[
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    LineAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
]
PredictionTypes = Union[
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    LinePrediction,
    PolygonPrediction,
    SegmentationPrediction,
]


class AnnotationOrPredictionFilter(NamedTuple):
    """Internal type for reconstruction of JSON encoded payload. Type field decides if filter behaves like FieldFilter
    or MetadataFilter

    Attributes:
        key: key to compare with value
        op: :class:`FilterOp` or one of [">", ">=", "<", "<=", "=", "==", "!=", "in", "not in"] to define comparison
            with value field
        value: bool, str, float or int to compare the field with key or list of the same values for 'in' and 'not in'
            ops
        allow_missing: Allow missing field values. Will REMOVE the object with the missing field from the selection
        type: DO NOT USE. Internal type for serialization over the wire. Changing this will change the `NamedTuple`
            type as well.
    """

    key: str
    op: Union[FilterOp, str]
    value: FilterableTypes
    allow_missing: bool
    type: FilterType


class FieldFilter(NamedTuple):
    """Filter on standard field of AnnotationTypes or PredictionTypes

    Examples:
        FieldFilter("x", ">", 10) would pass every :class:`BoxAnnotation` with `x` attribute larger than 10
        FieldFilter("label", "in", ["car", "truck"]) would pass every :class:`BoxAnnotation` with `label`
            in ["car", "truck"]

    Attributes:
        key: key to compare with value
        op: :class:`FilterOp` or one of [">", ">=", "<", "<=", "=", "==", "!=", "in", "not in"] to define comparison
            with value field
        value: bool, str, float or int to compare the field with key or list of the same values for 'in' and 'not in'
            ops
        allow_missing: Allow missing field values. Will REMOVE the object with the missing field from the selection
        type: DO NOT USE. Internal type for serialization over the wire. Changing this will change the `NamedTuple`
            type as well.
    """

    key: str
    op: Union[FilterOp, str]
    value: FilterableTypes
    allow_missing: bool = False
    type: FilterType = FilterType.FIELD


class MetadataFilter(NamedTuple):
    """Filter on customer provided metadata associated with AnnotationTypes or PredictionTypes

    Attributes:
        key: key to compare with value
        op: :class:`FilterOp` or one of [">", ">=", "<", "<=", "=", "==", "!=", "in", "not in"] to define comparison
            with value field
        value: bool, str, float or int to compare the field with key or list of the same values for 'in' and 'not in'
            ops
        allow_missing: Allow missing metadata values. Will REMOVE the object with the missing field from the selection
        type: DO NOT USE. Internal type for serialization over the wire. Changing this will change the `NamedTuple`
            type as well.
    """

    key: str
    op: Union[FilterOp, str]
    value: FilterableTypes
    allow_missing: bool = False
    type: FilterType = FilterType.METADATA


class SegmentMetadataFilter(NamedTuple):
    """Filter on customer provided metadata associated with Segments of a SegmentationAnnotation or
    SegmentationPrediction

    Attributes:
        key: key to compare with value
        op: :class:`FilterOp` or one of [">", ">=", "<", "<=", "=", "==", "!=", "in", "not in"] to define comparison
            with value field
        value: bool, str, float or int to compare the field with key or list of the same values for 'in' and 'not in'
            ops
        allow_missing: Allow missing metadata values. Will REMOVE the object with the missing field from the selection
        type: DO NOT USE. Internal type for serialization over the wire. Changing this will change the `NamedTuple`
            type as well.
    """

    key: str
    op: Union[FilterOp, str]
    value: FilterableTypes
    allow_missing: bool = False
    type: FilterType = FilterType.SEGMENT_METADATA


class SegmentFieldFilter(NamedTuple):
    """Filter on standard field of Segment(s) of SegmentationAnnotation and SegmentationPrediction

    Examples:
        SegmentFieldFilter("label", "in", ["grass", "tree"]) would pass every :class:`Segment` of a
            :class:`SegmentationAnnotation or :class:`SegmentationPrediction`

    Attributes:
        key: key to compare with value
        op: :class:`FilterOp` or one of [">", ">=", "<", "<=", "=", "==", "!=", "in", "not in"] to define comparison
            with value field
        value: bool, str, float or int to compare the field with key or list of the same values for 'in' and 'not in'
            ops
        allow_missing: Allow missing field values. Will REMOVE the object with the missing field from the selection
        type: DO NOT USE. Internal type for serialization over the wire. Changing this will change the `NamedTuple`
            type as well.
    """

    key: str
    op: Union[FilterOp, str]
    value: FilterableTypes
    allow_missing: bool = False
    type: FilterType = FilterType.SEGMENT_FIELD


Filter = Union[
    FieldFilter,
    MetadataFilter,
    AnnotationOrPredictionFilter,
    SegmentFieldFilter,
    SegmentMetadataFilter,
]
OrAndDNFFilters = List[List[Filter]]
OrAndDNFFilters.__doc__ = """\
Disjunctive normal form (DNF) filters.
DNF allows arbitrary boolean logical combinations of single field predicates.
The innermost structures each describe a single field predicate.

The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column
predicate.

Finally, the most outer list combines these filters as a disjunction (OR).
"""
ListOfOrAndJSONSerialized = List[List[List]]
ListOfOrAndJSONSerialized.__doc__ = """\
JSON serialized form of DNFFilters. The innermost list has to be trivially expandable (*list) to a
:class:`AnnotationOrPredictionFilter`.

Disjunctive normal form (DNF) filters.
DNF allows arbitrary boolean logical combinations of single field predicates.
The innermost structures each describe a single field predicate.
    -The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column
     predicate.
    -Finally, the most outer list combines these filters as a disjunction (OR).


"""
ListOfOrAndFilters = Union[OrAndDNFFilters, ListOfOrAndJSONSerialized]
ListOfAndJSONSerialized = List[List]
ListOfAndFilterTuple = List[Filter]
ListOfAndFilterTuple.__doc__ = """\
List of AND filters.
The list of predicates is interpreted as a conjunction (AND), forming a multiple field predicate.

If providing a doubly nested list the innermost list has to be trivially expandable (*list) to a
:class:`AnnotationOrPredictionFilter`
"""
ListOfAndFilters = Union[
    ListOfAndFilterTuple,
    ListOfAndJSONSerialized,
]

DNFFieldOrMetadataFilters = List[
    List[Union[FieldFilter, MetadataFilter, AnnotationOrPredictionFilter]]
]
DNFFieldOrMetadataFilters.__doc__ = """\
Disjunctive normal form (DNF) filters.
DNF allows arbitrary boolean logical combinations of single field predicates.
The innermost structures each describe a single field predicate.
-The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column
predicate.
"""


def _attribute_getter(
    field_name: str,
    allow_missing: bool,
    ann_or_pred: Union[AnnotationTypes, PredictionTypes, Segment],
):
    """Create a function to get object fields"""
    if allow_missing:
        return (
            getattr(ann_or_pred, field_name)
            if hasattr(ann_or_pred, field_name)
            else AlwaysFalseComparison()
        )
    else:
        return getattr(ann_or_pred, field_name)


class AlwaysFalseComparison:
    """Helper class to make sure that allow filtering out missing fields (by always returning a false comparison)"""

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return False


def _metadata_field_getter(
    field_name: str,
    allow_missing: bool,
    ann_or_pred: Union[AnnotationTypes, PredictionTypes, Segment],
):
    """Create a function to get a metadata field"""
    if isinstance(
        ann_or_pred, (SegmentationAnnotation, SegmentationPrediction)
    ):
        if allow_missing:
            logging.warning(
                "Trying to filter metadata on SegmentationAnnotation or Prediction. "
                "This will never work until metadata is supported for this format."
            )
            return AlwaysFalseComparison()
        else:
            raise RuntimeError(
                f"{type(ann_or_pred)} doesn't support metadata filtering"
            )

    if allow_missing:
        return (
            ann_or_pred.metadata.get(field_name, AlwaysFalseComparison())
            if ann_or_pred.metadata
            else AlwaysFalseComparison()
        )
    else:
        return (
            ann_or_pred.metadata[field_name]
            if ann_or_pred.metadata
            else RuntimeError(
                f"No metadata on {ann_or_pred}, trying to access {field_name}"
            )
        )


def _filter_to_comparison_function(  # pylint: disable=too-many-return-statements
    filter_def: Filter,
) -> Callable[[Union[AnnotationTypes, PredictionTypes, Segment]], bool]:
    """Creates a comparison function from a filter configuration to apply to annotations or predictions

    Parameters:
        filter_def: Definition of a filter conditions

    Returns:

    """
    if FilterType(filter_def.type) == FilterType.FIELD:
        getter = functools.partial(
            _attribute_getter, filter_def.key, filter_def.allow_missing
        )
    elif FilterType(filter_def.type) == FilterType.METADATA:
        getter = functools.partial(
            _metadata_field_getter, filter_def.key, filter_def.allow_missing
        )
    else:
        raise NotImplementedError(
            f"Unhandled filter type: {filter_def.type}. NOTE: Segmentation filters are handled elsewhere."
        )
    op = FilterOp(filter_def.op)
    if op is FilterOp.GT:
        return lambda ann_or_pred: getter(ann_or_pred) > filter_def.value
    elif op is FilterOp.GTE:
        return lambda ann_or_pred: getter(ann_or_pred) >= filter_def.value
    elif op is FilterOp.LT:
        return lambda ann_or_pred: getter(ann_or_pred) < filter_def.value
    elif op is FilterOp.LTE:
        return lambda ann_or_pred: getter(ann_or_pred) <= filter_def.value
    elif op is FilterOp.EQ or op is FilterOp.EQEQ:
        return lambda ann_or_pred: getter(ann_or_pred) == filter_def.value
    elif op is FilterOp.NEQ:
        return lambda ann_or_pred: getter(ann_or_pred) != filter_def.value
    elif op is FilterOp.IN:
        return lambda ann_or_pred: getter(ann_or_pred) in set(
            filter_def.value  # type: ignore
        )
    elif op is FilterOp.NOT_IN:
        return lambda ann_or_pred: getter(ann_or_pred) not in set(
            filter_def.value  # type:ignore
        )
    else:
        raise RuntimeError(
            f"Fell through all op cases, no match for: '{op}' - MetadataFilter: {filter_def},"
        )


def _apply_field_or_metadata_filters(
    filterable_sequence: Union[
        Sequence[AnnotationTypes], Sequence[PredictionTypes], Sequence[Segment]
    ],
    filters: DNFFieldOrMetadataFilters,
):
    """Apply filters to list of annotations or list of predictions or to a list of segments

    Attributes:
        filterable_sequence: Prediction, Annotation or Segment sequence
        filters: Filter predicates. Allowed formats are:
            ListOfAndFilters where each Filter forms a chain of AND predicates.
            or
            ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
            [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
            DNF allows arbitrary boolean logical combinations of single field
            predicates. The innermost structures each describe a single column predicate. The list of inner predicates
            is interpreted as a conjunction (AND), forming a more selective `and` multiple column predicate.
            Finally, the most outer list combines these filters as a disjunction (OR).
    """
    dnf_condition_functions = []
    for or_branch in filters:
        and_conditions = [
            _filter_to_comparison_function(cond) for cond in or_branch
        ]
        dnf_condition_functions.append(and_conditions)

    filtered = []
    for item in filterable_sequence:
        for or_conditions in dnf_condition_functions:
            if all(c(item) for c in or_conditions):
                filtered.append(item)
                break

    return filtered


def _split_segment_filters(
    dnf_filters: OrAndDNFFilters,
) -> Tuple[OrAndDNFFilters, OrAndDNFFilters]:
    """We treat Segment* filters differently -> this splits filters into two sets, one containing the
    standard field, metadata branches and the other the segment filters.
    """
    normal_or_branches = []
    segment_or_branches = []
    for and_branch in dnf_filters:
        normal_filters = []
        segment_filters = []
        for filter_statement in and_branch:
            if filter_statement.type in {
                FilterType.SEGMENT_METADATA,
                FilterType.SEGMENT_FIELD,
            }:
                segment_filters.append(filter_statement)
            else:
                normal_filters.append(filter_statement)
        normal_or_branches.append(normal_filters)
        segment_or_branches.append(segment_filters)
    return normal_or_branches, segment_or_branches


def _filter_segments(
    anns_or_preds: Union[
        Sequence[SegmentationAnnotation], Sequence[SegmentationPrediction]
    ],
    segment_filters: OrAndDNFFilters,
):
    """Filter Segments of a SegmentationAnnotation or Prediction

    We have to treat this differently as metadata and labels are on nested Segment objects
    """
    if len(segment_filters) == 0 or len(segment_filters[0]) == 0:
        return anns_or_preds

    # Transform segment filter types to field and metadata to iterate over annotation sub fields
    transformed_or_branches = (
        []
    )  # type: List[List[Union[MetadataFilter, FieldFilter]]]
    for and_branch in segment_filters:
        transformed_and = []  # type: List[Union[MetadataFilter, FieldFilter]]
        for filter_statement in and_branch:
            if filter_statement.type == FilterType.SEGMENT_FIELD:
                transformed_and.append(
                    FieldFilter(
                        filter_statement.key,
                        filter_statement.op,
                        filter_statement.value,
                        filter_statement.allow_missing,
                    )
                )
            elif filter_statement.type == FilterType.SEGMENT_METADATA:
                transformed_and.append(
                    MetadataFilter(
                        filter_statement.key,
                        filter_statement.op,
                        filter_statement.value,
                        filter_statement.allow_missing,
                    )
                )
            else:
                raise RuntimeError("Encountered a non SEGMENT_* filter type")

        transformed_or_branches.append(transformed_and)

    segments_filtered = []
    for ann_or_pred in anns_or_preds:
        if isinstance(
            ann_or_pred, (SegmentationAnnotation, SegmentationPrediction)
        ):
            ann_or_pred.annotations = _apply_field_or_metadata_filters(
                ann_or_pred.annotations, transformed_or_branches  # type: ignore
            )
            segments_filtered.append(ann_or_pred)

    return segments_filtered


def apply_filters(
    ann_or_pred: Union[Sequence[AnnotationTypes], Sequence[PredictionTypes]],
    filters: Union[ListOfOrAndFilters, ListOfAndFilters],
):
    """Apply filters to list of annotations or list of predictions
    Attributes:
        ann_or_pred: Prediction or Annotation
        filters: Filter predicates. Allowed formats are:
            ListOfAndFilters where each Filter forms a chain of AND predicates.
            or
            ListOfOrAndFilters where Filters are expressed in disjunctive normal form (DNF), like
            [[MetadataFilter("short_haired", "==", True), FieldFilter("label", "in", ["cat", "dog"]), ...].
            DNF allows arbitrary boolean logical combinations of single field
            predicates. The innermost structures each describe a single column predicate. The list of inner predicates
            is interpreted as a conjunction (AND), forming a more selective `and` multiple column predicate.
            Finally, the most outer list combines these filters as a disjunction (OR).
    """
    if filters is None or len(filters) == 0:
        return ann_or_pred

    dnf_filters = ensureDNFFilters(filters)
    filters, segment_filters = _split_segment_filters(dnf_filters)
    filtered = _apply_field_or_metadata_filters(ann_or_pred, filters)  # type: ignore
    filtered = _filter_segments(filtered, segment_filters)

    return filtered


def ensureDNFFilters(filters) -> OrAndDNFFilters:
    """JSON encoding creates a triple nested lists from the doubly nested tuples. This function creates the
    tuple form again."""
    if isinstance(
        filters[0],
        (
            MetadataFilter,
            FieldFilter,
            AnnotationOrPredictionFilter,
            SegmentFieldFilter,
            SegmentMetadataFilter,
        ),
    ):
        # Normalize into DNF
        filters: ListOfOrAndFilters = [filters]  # type: ignore

    # NOTE: We have to handle JSON transformed tuples which become two or three layers of lists
    if (
        isinstance(filters, list)
        and isinstance(filters[0], list)
        and isinstance(filters[0][0], str)
    ):
        filters = [filters]
    if (
        isinstance(filters, list)
        and isinstance(filters[0], list)
        and isinstance(filters[0][0], list)
    ):
        formatted_filter = []
        for or_branch in filters:
            and_chain = [
                AnnotationOrPredictionFilter(*condition)
                for condition in or_branch
            ]
            formatted_filter.append(and_chain)
        filters = formatted_filter
    return filters


def pretty_format_filters_with_or_and(
    filters: Optional[Union[ListOfOrAndFilters, ListOfAndFilters]]
):
    if filters is None:
        return "No filters applied!"
    dnf_filters = ensureDNFFilters(filters)
    or_branches = []
    for or_branch in dnf_filters:
        and_statements = []
        for and_branch in or_branch:
            if and_branch.type == FilterType.FIELD:
                class_name = "FieldFilter"
            elif and_branch.type == FilterType.METADATA:
                class_name = "MetadataFilter"
            elif and_branch.type == FilterType.SEGMENT_FIELD:
                class_name = "SegmentFieldFilter"
            elif and_branch.type == FilterType.SEGMENT_METADATA:
                class_name = "SegmentMetadataFilter"
            else:
                raise RuntimeError(
                    f"Un-handled filter type: {and_branch.type}"
                )
            op = (
                and_branch.op.value
                if isinstance(and_branch.op, FilterOp)
                else and_branch.op
            )
            value_formatted = (
                f'"{and_branch.value}"'
                if isinstance(and_branch.value, str)
                else f"{and_branch.value}".replace("'", '"')
            )
            statement = (
                f'{class_name}("{and_branch.key}", "{op}", {value_formatted})'
            )
            and_statements.append(statement)

        or_branches.append(and_statements)

    and_to_join = []
    for and_statements in or_branches:
        joined_and = " and ".join(and_statements)
        if len(or_branches) > 1 and len(and_statements) > 1:
            joined_and = "(" + joined_and + ")"
        and_to_join.append(joined_and)

    full_statement = " or ".join(and_to_join)
    return full_statement


def compose_helpful_filtering_error(
    ann_or_pred_list: Union[AnnotationList, PredictionList], filters
) -> List[str]:
    prefix = (
        "Annotations"
        if isinstance(ann_or_pred_list, AnnotationList)
        else "Predictions"
    )
    msg = []
    msg.append(f"{prefix}: All items filtered out by:")
    msg.append(f" {pretty_format_filters_with_or_and(filters)}")
    msg.append("")
    console = Console()
    table = Table(
        "Type",
        "Count",
        "Labels",
        title=f"Original {prefix}",
        title_justify="left",
    )
    for ann_or_pred_type, items in ann_or_pred_list.items():
        if items and isinstance(
            items[-1], (SegmentationAnnotation, SegmentationPrediction)
        ):
            labels = set()
            for seg in items:
                labels.update(set(s.label for s in seg.annotations))
        else:
            labels = set(a.label for a in items)
        if items:
            table.add_row(ann_or_pred_type, str(len(items)), str(list(labels)))
    with console.capture() as capture:
        console.print(table)
    msg.append(capture.get())
    return msg


def filter_annotation_list(
    annotations: AnnotationList, annotation_filters
) -> AnnotationList:
    annotations = copy.deepcopy(annotations)
    if annotation_filters is None or len(annotation_filters) == 0:
        return annotations
    annotations.box_annotations = apply_filters(
        annotations.box_annotations, annotation_filters
    )
    annotations.line_annotations = apply_filters(
        annotations.line_annotations, annotation_filters
    )
    annotations.polygon_annotations = apply_filters(
        annotations.polygon_annotations, annotation_filters
    )
    annotations.cuboid_annotations = apply_filters(
        annotations.cuboid_annotations, annotation_filters
    )
    annotations.category_annotations = apply_filters(
        annotations.category_annotations, annotation_filters
    )
    annotations.multi_category_annotations = apply_filters(
        annotations.multi_category_annotations, annotation_filters
    )
    annotations.segmentation_annotations = apply_filters(
        annotations.segmentation_annotations, annotation_filters
    )
    return annotations


def filter_prediction_list(
    predictions: PredictionList, prediction_filters
) -> PredictionList:
    predictions = copy.deepcopy(predictions)
    if prediction_filters is None or len(prediction_filters) == 0:
        return predictions
    predictions.box_predictions = apply_filters(
        predictions.box_predictions, prediction_filters
    )
    predictions.line_predictions = apply_filters(
        predictions.line_predictions, prediction_filters
    )
    predictions.polygon_predictions = apply_filters(
        predictions.polygon_predictions, prediction_filters
    )
    predictions.cuboid_predictions = apply_filters(
        predictions.cuboid_predictions, prediction_filters
    )
    predictions.category_predictions = apply_filters(
        predictions.category_predictions, prediction_filters
    )
    predictions.segmentation_predictions = apply_filters(
        predictions.segmentation_predictions, prediction_filters
    )
    return predictions
