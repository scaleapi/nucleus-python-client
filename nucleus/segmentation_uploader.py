# import asyncio
# import json
# import os
# from typing import TYPE_CHECKING, Any, List
# from nucleus.annotation import SegmentationAnnotation
# from nucleus.async_utils import get_event_loop
# from nucleus.constants import DATASET_ID_KEY, MASK_TYPE, SEGMENTATIONS_KEY
# from nucleus.errors import NotFoundError
# from nucleus.payload_constructor import construct_segmentation_payload
# from annotation import is_local_path
# from nucleus.upload_response import UploadResponse
# import nest_asyncio

# if TYPE_CHECKING:
#     from . import NucleusClient


# class SegmentationUploader:
#     def __init__(self, dataset_id: str, client: "NucleusClient"):  # noqa: F821
#         self.dataset_id = dataset_id
#         self._client = client

#     def annotate(
#         self,
#         segmentations: List[SegmentationAnnotation],
#         batch_size: int = 20,
#         update: bool = False,
#     ):
#         remote_segmentations = []
#         local_segmentations = []
#         for segmentation in segmentations:
#             if is_local_path(segmentation.mask_url):
#                 if not segmentation.local_file_exists():
#                     raise NotFoundError(
#                         "Could not find f{segmentation.mask_url}"
#                     )
#                 local_segmentations.append(segmentation)
#             else:
#                 remote_segmentations.append(segmentation)

#         local_batches = [
#             local_segmentations[i : i + batch_size]
#             for i in range(0, len(local_segmentations), batch_size)
#         ]

#         remote_batches = [
#             remote_segmentations[i : i + batch_size]
#             for i in range(0, len(remote_segmentations), batch_size)
#         ]

#         agg_response = UploadResponse(json={DATASET_ID_KEY: self.dataset_id})

#         async_responses: List[Any] = []

#         if local_batches:
#             tqdm_local_batches = self._client.tqdm_bar(
#                 local_batches, desc="Local file batches"
#             )
#             for batch in tqdm_local_batches:
#                 responses = self._process_annotate_requests_local(
#                     self.dataset_id, batch
#                 )
#                 async_responses.extend(responses)

#         def process_annotate_requests_local(
#             dataset_id: str,
#             segmentations: List[SegmentationAnnotation],
#             local_batch_size: int = 10,
#         ):
#             requests = []
#             file_pointers = []
#             for i in range(0, len(segmentations), local_batch_size):
#                 batch = segmentations[i : i + local_batch_size]
#                 request, request_file_pointers = self.construct_files_request(
#                     batch
#                 )
#                 requests.append(request)
#                 file_pointers.extend(request_file_pointers)

#             future = self.make_many_files_requests_asynchronously(
#                 requests, f"dataset/{dataset_id}/files"
#             )

#             loop = get_event_loop()

#             responses = loop.run_until_complete(future)
#             [fp.close() for fp in file_pointers]
#             return responses

#         def construct_files_request(
#             segmentations: List[SegmentationAnnotation],
#         ):
#             request_json = construct_segmentation_payload(
#                 segmentations, update
#             )
#             request_payload = [
#                 (
#                     SEGMENTATIONS_KEY,
#                     (None, json.dumps(request_json), "application/json"),
#                 )
#             ]
#             file_pointers = []
#             for segmentation in segmentations:
#                 mask_fp = open(segmentation.mask_url, "rb")
#                 filename = os.path.basename(segmentation.mask_url)
#                 file_type = segmentation.mask_url.split(".")[-1]
#                 if file_type != "png":
#                     raise ValueError(
#                         f"Only png files are supported. Got {file_type} for {segmentation.mask_url}"
#                     )
#                 request_payload.append(
#                     (MASK_TYPE, (filename, mask_fp, "image/png"))
#                 )
#             return request_payload, file_pointers


# {"items": [{"metadata": {"test": 0}, "reference_id": "test_img.jpg", "image_url": "tests/test_img.jpg", "upload_to_scale": true}]

# [{"metadata": {"test": 0}, "reference_id": "test_img.jpg", "image_url": "tests/test_img.jpg", "upload_to_scale": true, "update": false}]
