from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import NucleusClient


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
