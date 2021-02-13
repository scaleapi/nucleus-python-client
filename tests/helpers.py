from pathlib import Path

TEST_MODEL_NAME = '[PyTest] Test Model'
TEST_MODEL_REFERENCE = '[PyTest] Test Model Reference'
TEST_MODEL_RUN = '[PyTest] Test Model Run'
TEST_DATASET_NAME = '[PyTest] Test Dataset'
TEST_IMG_URLS = [
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/6dd63871-831611a6.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/82c1005c-e2d1d94f.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/7f2e1814-6591087d.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/06924f46-1708b96f.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/89b42832-10d662f4.jpg',
]

def reference_id_from_url(url):
    return Path(url).name

TEST_BOX_ANNOTATIONS = [
    {
        'label': f'[Pytest] Box Annotation ${i}',
        'x': 50 + i * 10,
        'y': 60 + i * 10,
        'width': 70 + i * 10,
        'height': 80 + i * 10,
        'reference_id': reference_id_from_url(TEST_IMG_URLS[i]),
        'annotation_id': f'[Pytest] Box Annotation Annotation Id{i}',
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_POLYGON_ANNOTATIONS = [
    {
        'label': f'[Pytest] Polygon Annotation ${i}',
        'vertices': [
            {
                'x': 50 + i * 10 + j,
                'y': 60 + i * 10 + j,
            }
            for j in range(3)
        ],
        'reference_id': reference_id_from_url(TEST_IMG_URLS[i]),
        'annotation_id': f'[Pytest] Polygon Annotation Annotation Id{i}',
    }
    for i in range(len(TEST_IMG_URLS))
]

TEST_BOX_PREDICTIONS = [
    {
        **TEST_BOX_ANNOTATIONS[i],
        'confidence': 0.10 * i
    }
    for i in range(len(TEST_BOX_ANNOTATIONS))
]

TEST_POLYGON_PREDICTIONS = [
    {
        **TEST_POLYGON_ANNOTATIONS[i],
        'confidence': 0.10 * i
    }
    for i in range(len(TEST_POLYGON_ANNOTATIONS))
]


# Asserts that a box annotation instance matches a dict representing its properties.
# Useful to check annotation uploads/updates match.
def assert_box_annotation_matches_dict(annotation_instance, annotation_dict):
    assert annotation_instance.label == annotation_dict['label']
    assert annotation_instance.x == annotation_dict['x']
    assert annotation_instance.y == annotation_dict['y']
    assert annotation_instance.height == annotation_dict['height']
    assert annotation_instance.width == annotation_dict['width']
    assert annotation_instance.annotation_id == annotation_dict['annotation_id']

def assert_polygon_annotation_matches_dict(annotation_instance, annotation_dict):
    assert annotation_instance.label == annotation_dict['label']
    assert annotation_instance.annotation_id == annotation_dict['annotation_id']
    for instance_pt, dict_pt in zip(annotation_instance.vertices, annotation_dict['vertices']):
        assert instance_pt['x'] == dict_pt['x']
        assert instance_pt['y'] == dict_pt['y']

# Asserts that a box prediction instance matches a dict representing its properties.
# Useful to check prediction uploads/updates match.
def assert_box_prediction_matches_dict(prediction_instance, prediction_dict):
    assert_box_annotation_matches_dict(prediction_instance, prediction_dict)
    assert prediction_instance.confidence == prediction_dict['confidence']

def assert_polygon_prediction_matches_dict(prediction_instance, prediction_dict):
    assert_polygon_annotation_matches_dict(prediction_instance, prediction_dict)
    assert prediction_instance.confidence == prediction_dict['confidence']
