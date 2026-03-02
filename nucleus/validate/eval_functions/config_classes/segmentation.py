from typing import Optional, Union

from nucleus.validate.eval_functions.base_eval_function import (
    EvalFunctionConfig,
)

from ....metrics.filtering import ListOfAndFilters, ListOfOrAndFilters


class SegmentationIOUConfig(EvalFunctionConfig):
    def __call__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`SegmentationIOU` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            poly_iou: BoundingBoxIOU = client.validate.eval_functions.seg_iou
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[poly_iou(confidence_threshold=0.8) > 0.5]
            )

        Args:
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "seg_iou"


class SegmentationMAPConfig(EvalFunctionConfig):
    def __call__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`SegmentationMAP` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            seg_map: SegmentationMAP= client.validate.eval_functions.seg_map
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[seg_map(iou_threshold=0.6) > 0.8]
            )

        Args:
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "seg_map"


class SegmentationRecallConfig(EvalFunctionConfig):
    def __call__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`SegmentationRecall` object.
        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            seg_recall = client.validate.eval_functions.seg_recall
            slice_id = "slc_<your_slice>"
            scenario_test = client.validate.create_scenario_test(
                "Example test",
                slice_id=slice_id,
                evaluation_criteria=[seg_recall(, confidence_threshold=0.4) > 0.9]
            )

        Args:
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "seg_recall"


class SegmentationPrecisionConfig(EvalFunctionConfig):
    def __call__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Configures a call to :class:`SegmentationPrecision` object.
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "seg_precision"


class SegmentationFWAVACCConfig(EvalFunctionConfig):
    def __call__(
        self,
        annotation_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        prediction_filters: Optional[
            Union[ListOfOrAndFilters, ListOfAndFilters]
        ] = None,
        **kwargs,
    ):
        """Initializes SegmentationFWAVACC object.

        Args:
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
            annotation_filters=annotation_filters,
            prediction_filters=prediction_filters,
            **kwargs,
        )

    @classmethod
    def expected_name(cls) -> str:
        return "seg_fwavacc"
