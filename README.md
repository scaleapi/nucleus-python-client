# Nucleus

## Installation

### Editable mode

`$ pip install -e . `

### As a Normal Package

`$ pip install git+ssh://git@github.com/scaleapi/nucleus-python-client.git`

## Usage

```python
import nucleus

client = nucleus.NucleusClient('YOUR_API_KEY_HERE')
```

### Create Dataset

```python
response = client.create_dataset({"name": "My Dataset"})
dataset = client.get_dataset(response["dataset_id"])
```

### Append Items to a Dataset
You can append both local images and images from the web.
Each image object is a dictionary with three fields:
```python
datasetItem1 = {'image_url': 'http://<my_image_url>', 'reference_id': 'my_image_name.jpg',
  'metadata': {'label': '0'}}
```

The append function expects a list of datasetItems to upload, like this:
```python
response = dataset.append([datasetItem1])
```

If you're uploading a local image, you can specify a filepath as the image_url.
```python
datasetItem2 = {'image_url': './data_folder/my_img_001.png', 'reference_id': 'my_img_001.png',
  'metadata': {'label': '1'}}
response = dataset.append([datasetItem2], local = True)
```

### Access Dataset Items
DatasetItems are accessible by reference id
```python
# returns datasetItem2 (above)
item = dataset.refloc('my_img_001.png')
```

### Add Annotations
Box2DAnnotation has same format as https://dashboard.scale.com/nucleus/docs/api#add-ground-truth
```python
response = dataset.annotate({'annotations:' [Box2DAnnotation, ..., Box2DAnnotation]});
```

### Get Dataset Info
Tells us the dataset name, number of dataset items, model_runs, and slice_ids.
```python
dataset.info()
```

### Add Model

```python
response = client.add_model({"name": "My Model", "reference_id": "model-0.5", "metadata": {"iou_thr": 0.5}})
```

### Add Model Run

```python
dataset.create_model_run({"reference_id": "model-0.5"})
```
