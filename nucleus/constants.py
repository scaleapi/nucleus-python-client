ANNOTATIONS_IGNORED_KEY = "annotations_ignored"
ANNOTATIONS_KEY = "annotations"
ANNOTATIONS_PROCESSED_KEY = "annotations_processed"
ANNOTATION_ID_KEY = "annotation_id"
ANNOTATION_METADATA_SCHEMA_KEY = "annotation_metadata_schema"
BACKFILL_JOB_KEY = "backfill_job"
BOX_TYPE = "box"
LINE_TYPE = "line"
POLYGON_TYPE = "polygon"
MASK_TYPE = "mask"
SEGMENTATION_TYPE = "segmentation"
CUBOID_TYPE = "cuboid"
KEYPOINTS_TYPE = "keypoints"
CATEGORY_TYPE = "category"
MULTICATEGORY_TYPE = "multicategory"
ANNOTATION_TYPES = (
    BOX_TYPE,
    LINE_TYPE,
    POLYGON_TYPE,
    KEYPOINTS_TYPE,
    SEGMENTATION_TYPE,
    CUBOID_TYPE,
    CATEGORY_TYPE,
    MULTICATEGORY_TYPE,
)
ANNOTATION_UPDATE_KEY = "update"
AUTOTAGS_KEY = "autotags"
AUTOTAG_SCORE_THRESHOLD = "score_threshold"
EXPORTED_ROWS = "exportedRows"
EXPORTED_SCALE_TASK_INFO_ROWS = "exportedScaleTaskInfoRows"
CAMERA_MODEL_KEY = "camera_model"
CAMERA_PARAMS_KEY = "camera_params"
CLASS_PDF_KEY = "class_pdf"
CONFIDENCE_KEY = "confidence"
CX_KEY = "cx"
CY_KEY = "cy"
DATASET_ID_KEY = "dataset_id"
DATASET_IS_SCENE_KEY = "is_scene"
DATASET_ITEM_ID_KEY = "dataset_item_id"
DATASET_LENGTH_KEY = "length"
DATASET_MODEL_RUNS_KEY = "model_run_ids"
DATASET_NAME_KEY = "name"
DATASET_SLICES_KEY = "slice_ids"
DEFAULT_ANNOTATION_UPDATE_MODE = False
DEFAULT_NETWORK_TIMEOUT_SEC = 120
DIMENSIONS_KEY = "dimensions"
EMBEDDING_VECTOR_KEY = "embedding_vector"
EMBEDDINGS_URL_KEY = "embeddings_urls"
EMBEDDING_DIMENSION_KEY = "embedding_dimension"
ERRORS_KEY = "errors"
ERROR_CODES = "error_codes"
ERROR_ITEMS = "upload_errors"
ERROR_PAYLOAD = "error_payload"
FRAME_RATE_KEY = "frame_rate"
FRAMES_KEY = "frames"
FX_KEY = "fx"
FY_KEY = "fy"
GEOMETRY_KEY = "geometry"
HEADING_KEY = "heading"
HEIGHT_KEY = "height"
ID_KEY = "id"
IGNORED_ITEMS = "ignored_items"
IMAGE_KEY = "image"
IMAGE_LOCATION_KEY = "image_location"
IMAGE_URL_KEY = "image_url"
INDEX_KEY = "index"
INDEX_CONTINUOUS_ENABLE_KEY = "enable"
ITEMS_KEY = "items"
ITEM_KEY = "item"
ITEM_METADATA_SCHEMA_KEY = "item_metadata_schema"
JOB_ID_KEY = "job_id"
K1_KEY = "k1"
K2_KEY = "k2"
K3_KEY = "k3"
K4_KEY = "k4"
KEEP_HISTORY_KEY = "keep_history"
LENGTH_KEY = "length"
JOB_STATUS_KEY = "job_status"
JOB_LAST_KNOWN_STATUS_KEY = "job_last_known_status"
JOB_TYPE_KEY = "job_type"
JOB_CREATION_TIME_KEY = "job_creation_time"
KEYPOINTS_KEY = "keypoints"
KEYPOINTS_NAMES_KEY = "names"
KEYPOINTS_SKELETON_KEY = "skeleton"
LAST_PAGE = "lastPage"
LABEL_KEY = "label"
LABELS_KEY = "labels"
MASK_URL_KEY = "mask_url"
MAX_PAYLOAD_SIZE = 0x1FFFFFE8  # Set to max string size since we currently convert payloads to strings for processing on the server-side
MESSAGE_KEY = "message"
METADATA_KEY = "metadata"
MODEL_BUNDLE_NAME_KEY = "bundle_name"
MODEL_TAGS_KEY = "tags"
MODEL_ID_KEY = "model_id"
MODEL_RUN_ID_KEY = "model_run_id"
NAME_KEY = "name"
NEW_ITEMS = "new_items"
NUCLEUS_ENDPOINT = "https://api.scale.com/v1/nucleus"
NUM_SENSORS_KEY = "num_sensors"
ORIGINAL_IMAGE_URL_KEY = "original_image_url"
PAGE_SIZE = "pageSize"
PAGE_TOKEN = "pageToken"
P1_KEY = "p1"
P2_KEY = "p2"
POINTCLOUD_KEY = "pointcloud"
POINTCLOUD_LOCATION_KEY = "pointcloud_location"
POINTCLOUD_URL_KEY = "pointcloud_url"
POSITION_KEY = "position"
PREDICTIONS_IGNORED_KEY = "predictions_ignored"
PREDICTIONS_KEY = "predictions"
PREDICTIONS_PROCESSED_KEY = "predictions_processed"
REFERENCE_IDS_KEY = "reference_ids"
REFERENCE_ID_KEY = "reference_id"
BACKEND_REFERENCE_ID_KEY = "ref_id"  # TODO(355762): Our backend returns this instead of the "proper" key sometimes.
REQUEST_ID_KEY = "requestId"
SCALE_TASK_INFO_KEY = "scale_task_info"
SCENE_KEY = "scene"
SCENES_KEY = "scenes"
SERIALIZED_REQUEST_KEY = "serialized_request"
SEGMENTATIONS_KEY = "segmentations"
SLICE_ID_KEY = "slice_id"
STATUS_CODE_KEY = "status_code"
STATUS_KEY = "status"
SUCCESS_STATUS_CODES = [200, 201, 202]
TAXONOMY_NAME_KEY = "taxonomy_name"
TYPE_KEY = "type"
UPDATED_ITEMS = "updated_items"
UPDATE_KEY = "update"
UPLOAD_TO_SCALE_KEY = "upload_to_scale"
URL_KEY = "url"
VERTICES_KEY = "vertices"
VIDEO_LOCATION_KEY = "video_location"
VIDEO_URL_KEY = "video_url"
VISIBLE_KEY = "visible"
WIDTH_KEY = "width"
YAW_KEY = "yaw"
W_KEY = "w"
X_KEY = "x"
Y_KEY = "y"
Z_KEY = "z"
