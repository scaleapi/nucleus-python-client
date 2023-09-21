from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List

from .async_job import AsyncJob


class CustomerJobTypes(str, Enum):
    UPLOAD_DATASET_ITEMS = "uploadDatasetItems"
    UPLOAD_PREDICTIONS = "uploadPredictions"
    UPLOAD_ANNOTATIONS = "uploadAnnotations"
    UPLOAD_LIDAR_SCENE = "uploadLidarScene"
    UPLOAD_VIDEO_SCENE = "uploadVideoScene"
    MODEL_INFERENCE_RUN = "modelInferenceRun"
    INDEXING_IMAGE = "indexingImage"
    INDEXING_OBJECT = "indexingObject"
    ANNOTATION_DELETION = "annotationDeletion"
    SEND_TO_LABELING = "sendToLabeling"
    INGEST_TASKS = "ingestTasks"
    CUSTOM_INDEXING = "customIndexing"
    EMBEDDING_DELETE = "embeddingDelete"
    TEST_EVALUATION = "testEvaluation"
    VALIDATE_METRICS = "modelMetrics"
    MODEL_RUN_COMMIT = "modelRunCommit"
    AUTOTAG_INFERENCE = "autotagInference"
    SLICE_QUERY = "sliceQuery"
    CLONE_DATASET = "cloneDataset"
    METADATA_UPDATE = "metadataUpdate"
    TRIGGER_EVALUATE = "triggerEvaluate"
    EXPORT_EMBEDDINGS = "exportEmbeddings"

    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        return True

    @staticmethod
    def options():
        return list(map(lambda c: c.value, CustomerJobTypes))


def jobs_status_overview(jobs: List[AsyncJob]) -> Dict[str, Any]:
    jobs_by_type = defaultdict(list)
    for job in jobs:
        jobs_by_type[job.job_type].append(job)

    jobs_status = {}
    for job_type, job_collection in jobs_by_type.items():
        overview = defaultdict(int)  # type: Dict[str, int]
        for job in job_collection:
            overview[job.job_last_known_status] += 1
        jobs_status[job_type] = dict(overview)

    return jobs_status
