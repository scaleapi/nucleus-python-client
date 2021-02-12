from pathlib import Path

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
