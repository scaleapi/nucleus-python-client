import json
from collections import Counter
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

from nucleus.annotation import Annotation, SegmentationAnnotation
from nucleus.async_utils import (
    FileFormField,
    FormDataContextHandler,
    make_many_form_data_requests_concurrently,
)
from nucleus.constants import MASK_TYPE, SERIALIZED_REQUEST_KEY
from nucleus.errors import DuplicateIDError
from nucleus.payload_constructor import (
    construct_annotation_payload,
    construct_segmentation_payload,
)

if TYPE_CHECKING:
    from . import NucleusClient


def accumulate_dict_values(dicts: Iterable[dict]):
    """
    Accumulate a list of dicts into a single dict using summation.
    """
    result = {}
    for d in dicts:
        for key, value in d.items():
            if (
                key not in result
                or key == "dataset_id"
                or key == "model_run_id"
            ):
                result[key] = value
            else:
                result[key] += value
    return result


class AnnotationUploader:
    """This is a helper class not intended for direct use. Please use dataset.annotate
    or dataset.upload_predictions.

    This class is purely a helper class for implementing dataset.annotate/dataset.predict.
    """

    def __init__(
        self, dataset_id: Optional[str], client: "NucleusClient"
    ):  # noqa: F821
        self._client = client
        self._route = f"dataset/{dataset_id}/annotate"

    def upload(
        self,
        annotations: Iterable[Annotation],
        batch_size: int = 5000,
        update: bool = False,
        remote_files_per_upload_request: int = 20,
        local_files_per_upload_request: int = 10,
        local_file_upload_concurrency: int = 30,
    ):
        """For more details on parameters and functionality, see dataset.annotate."""
        if local_files_per_upload_request > 10:
            raise ValueError("local_files_per_upload_request must be <= 10")
        annotations_without_files: List[Annotation] = []
        segmentations_with_local_files: List[SegmentationAnnotation] = []
        segmentations_with_remote_files: List[SegmentationAnnotation] = []

        for annotation in annotations:
            if annotation.has_local_files_to_upload():
                # Only segmentations have local files currently, and probably for a long
                # time to to come.
                assert isinstance(annotation, SegmentationAnnotation)
                segmentations_with_local_files.append(annotation)
            elif isinstance(annotation, SegmentationAnnotation):
                segmentations_with_remote_files.append(annotation)
            else:
                annotations_without_files.append(annotation)

        responses = []
        if segmentations_with_local_files:
            responses.extend(
                self.make_batched_file_form_data_requests(
                    segmentations=segmentations_with_local_files,
                    update=update,
                    local_files_per_upload_request=local_files_per_upload_request,
                    local_file_upload_concurrency=local_file_upload_concurrency,
                )
            )
        if segmentations_with_remote_files:
            # Segmentations require an upload and must be batched differently since a single
            # segmentation will take a lot longer for the server to process than a single
            # annotation of any other kind.
            responses.extend(
                self.make_batched_requests(
                    segmentations_with_remote_files,
                    update,
                    batch_size=remote_files_per_upload_request,
                    segmentation=True,
                )
            )
        if annotations_without_files:
            responses.extend(
                self.make_batched_requests(
                    annotations_without_files,
                    update,
                    batch_size=batch_size,
                    segmentation=False,
                )
            )

        return accumulate_dict_values(responses)

    def make_batched_requests(
        self,
        annotations: Sequence[Annotation],
        update: bool,
        batch_size: int,
        segmentation: bool,
    ):
        batches = [
            annotations[i : i + batch_size]
            for i in range(0, len(annotations), batch_size)
        ]
        responses = []
        progress_bar_name = (
            "Segmentation batches" if segmentation else "Annotation batches"
        )
        for batch in self._client.tqdm_bar(batches, desc=progress_bar_name):
            payload = construct_annotation_payload(batch, update)
            responses.append(
                self._client.make_request(payload, route=self._route)
            )
        return responses

    def make_batched_file_form_data_requests(
        self,
        segmentations: Sequence[SegmentationAnnotation],
        update,
        local_files_per_upload_request: int,
        local_file_upload_concurrency: int,
    ):
        requests = []
        for i in range(0, len(segmentations), local_files_per_upload_request):
            batch = segmentations[i : i + local_files_per_upload_request]
            request = FormDataContextHandler(
                self.get_form_data_and_file_pointers_fn(batch, update)
            )
            requests.append(request)

        progressbar = self._client.tqdm_bar(
            total=len(requests),
            desc="Local segmentation mask file batches",
        )

        return make_many_form_data_requests_concurrently(
            client=self._client,
            requests=requests,
            route=self._route,
            progressbar=progressbar,
            concurrency=local_file_upload_concurrency,
        )

    def get_form_data_and_file_pointers_fn(
        self,
        segmentations: Iterable[SegmentationAnnotation],
        update: bool,
    ):
        """Defines a function to be called on each retry.

        File pointers are also returned so whoever calls this function can
        appropriately close the files. This is intended for use with a
        FormDataContextHandler in order to make form data requests.
        """

        def fn():
            request_json = construct_segmentation_payload(
                segmentations, update
            )
            form_data = [
                FileFormField(
                    name=SERIALIZED_REQUEST_KEY,
                    filename=None,
                    value=json.dumps(request_json),
                    content_type="application/json",
                )
            ]
            file_pointers = []
            for segmentation in segmentations:
                # I don't know of a way to use with, since all files in the request
                # need to be opened at the same time.
                # pylint: disable=consider-using-with
                mask_fp = open(segmentation.mask_url, "rb")
                # pylint: enable=consider-using-with
                file_type = segmentation.mask_url.split(".")[-1]
                if file_type != "png":
                    raise ValueError(
                        f"Only png files are supported. Got {file_type} for {segmentation.mask_url}"
                    )
                form_data.append(
                    FileFormField(
                        name=MASK_TYPE,
                        filename=segmentation.mask_url,
                        value=mask_fp,
                        content_type="image/png",
                    )
                )
                file_pointers.append(mask_fp)
            return form_data, file_pointers

        return fn

    @staticmethod
    def check_for_duplicate_ids(annotations: Iterable[Annotation]):
        """Do not allow annotations to have the same (annotation_id, reference_id) tuple"""

        # some annotations like CategoryAnnotation do not have annotation_id attribute, and as such, we allow duplicates
        tuple_ids = [
            (ann.reference_id, ann.annotation_id)  # type: ignore
            for ann in annotations
            if hasattr(ann, "annotation_id")
        ]
        tuple_count = Counter(tuple_ids)
        duplicates = {key for key, value in tuple_count.items() if value > 1}
        if len(duplicates) > 0:
            raise DuplicateIDError(
                f"Duplicate annotations with the same (reference_id, annotation_id) properties found.\n"
                f"Duplicates: {duplicates}\n"
                f"To fix this, avoid duplicate annotations, or specify a different annotation_id attribute "
                f"for the failing items."
            )


class PredictionUploader(AnnotationUploader):
    def __init__(
        self,
        client: "NucleusClient",
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
        model_run_id: Optional[str] = None,
    ):
        super().__init__(dataset_id, client)
        self._client = client
        if model_run_id is not None:
            assert model_id is None and dataset_id is None
            self._route = f"modelRun/{model_run_id}/predict"
        else:
            assert (
                model_id is not None and dataset_id is not None
            ), "Model ID and dataset ID are required if not using model run id."
            self._route = (
                f"dataset/{dataset_id}/model/{model_id}/uploadPredictions"
            )
