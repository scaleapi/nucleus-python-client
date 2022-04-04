import enum
import functools
from enum import Enum
from typing import Callable, Iterable, List, NamedTuple, Sequence, Set, Union

from nucleus.annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    LineAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)
from nucleus.prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    LinePrediction,
    PolygonPrediction,
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
    """

    FIELD = "field"
    METADATA = "metadata"


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
        FieldFilter("label", "in", [) would pass every :class:`BoxAnnotation` with `x` attribute larger than 10

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
        allow_missing: Allow missing metada values. Will REMOVE the object with the missing field from the selection
        type: DO NOT USE. Internal type for serialization over the wire. Changing this will change the `NamedTuple`
            type as well.
    """

    key: str
    op: Union[FilterOp, str]
    value: FilterableTypes
    allow_missing: bool = False
    type: FilterType = FilterType.METADATA


Filter = Union[FieldFilter, MetadataFilter, AnnotationOrPredictionFilter]
DNFFilters = List[List[Filter]]
DNFFilters.__doc__ = """\
Disjunctive normal form (DNF) filters.
DNF allows arbitrary boolean logical combinations of single field predicates.
The innermost structures each describe a single field predicate.

The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column
predicate.

Finally, the most outer list combines these filters as a disjunction (OR).
"""
ListOfOrAndFilters = Union[DNFFilters, List[List[List]]]
ListOfOrAndFilters.__doc__ = """\
Disjunctive normal form (DNF) filters.
DNF allows arbitrary boolean logical combinations of single field predicates.
The innermost structures each describe a single field predicate.
    -The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column
     predicate.
    -Finally, the most outer list combines these filters as a disjunction (OR).

If providing a triple nested list the innermost list has to be trivially expandable (*list) to a
:class:`AnnotationOrPredictionFilter`
"""
ListOfAndFilters = Union[
    List[Filter],
    List[List],
]
ListOfAndFilters.__doc__ = """\
List of AND filters.
The list of predicates is interpreted as a conjunction (AND), forming a multiple field predicate.

If providing a doubly nested list the innermost list has to be trivially expandable (*list) to a
:class:`AnnotationOrPredictionFilter`
"""


def _attribute_getter(
    field_name: str,
    allow_missing: bool,
    ann_or_pred: Union[AnnotationTypes, PredictionTypes],
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
    ann_or_pred: Union[AnnotationTypes, PredictionTypes],
):
    """Create a function to get a metadata field"""
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
) -> Callable[[Union[AnnotationTypes, PredictionTypes]], bool]:
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

    filters = ensureDNFFilters(filters)

    dnf_condition_functions = []
    for or_branch in filters:
        and_conditions = [
            _filter_to_comparison_function(cond) for cond in or_branch
        ]
        dnf_condition_functions.append(and_conditions)

    filtered = []
    for item in ann_or_pred:
        for or_conditions in dnf_condition_functions:
            if all(c(item) for c in or_conditions):
                filtered.append(item)
                break
    return filtered


def ensureDNFFilters(filters) -> DNFFilters:
    """JSON encoding creates a triple nested lists from the doubly nested tuples. This function creates the
    tuple form again."""
    if isinstance(filters[0], (MetadataFilter, FieldFilter)):
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
