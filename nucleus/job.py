from enum import Enum
from typing import List


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

    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        return True

    @staticmethod
    def options():
        return list(map(lambda c: c.value, CustomerJobTypes))


class NucleusJob:
    def __init__(self):
        pass


class NucleusJobs:
    def __init__(self, client: "NucleusClient", dataset_id: str):
        self._client = client
        self.dataset_id = dataset_id

    def dataset_jobs_status(
        self, job_types, from_date, to_date
    ) -> List[NucleusJob]:

        payload = {
            job_types: job_types,
            from_date: from_date,
            to_date: to_date,
        }
        resp = self._client.make_request(
            payload=payload,
            route=f"dataset/{self.dataset_id}/jobs/statusasync={is_async}",
        )

        return []
