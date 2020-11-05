# Nucleus

## Installation

### Editable mode

`$ pip install -e . `

### As a Normal Package

`$ pip install git+ssh://git@github.com/scaleapi/nucleus-python-client.git`

## Usage

The first step to using the Nucleus library is instantiating a client object.
The client abstractions serves to authenticate the user and act as the gateway
for users to interact with their datasets, models, and model runs.

### Create a client object
```python
import nucleus
client = nucleus.NucleusClient("YOUR_API_KEY_HERE")
```

### Create Dataset
```python
response = client.create_dataset({"name": "My Dataset"})
dataset = client.get_dataset(response["dataset_id"])
```

### List Datasets
```python
datasets = client.list_datasets()
```

### Delete a Dataset
By specifying target dataset id.
A response code of 200 indicates successful deletion.
```python
client.delete_dataset("YOUR_DATASET_ID")
```

### Append Items to a Dataset
You can append both local images and images from the web.
Each image object is a dictionary with three fields:
```python
datasetItem1 = {"image_url": "http://<my_image_url>", "reference_id": "my_image_name.jpg",
  "metadata": {"label": "0"}}
```

The append function expects a list of datasetItems to upload, like this:
```python
response = dataset.append({"items": [datasetItem2]})
```

If you're uploading a local image, you can specify a filepath as the image_url.
```python
datasetItem2 = {"image_url": "./data_folder/my_img_001.png", "reference_id": "my_img_001.png",
  "metadata": {"label": "1"}}
response = dataset.append({"items": [datasetItem2]}, local = True)
```

For particularly large item uploads, consider using one of the example scripts located in **references**
These scripts upload items in batches for easier debugging.

### Get Dataset Info
Tells us the dataset name, number of dataset items, model_runs, and slice_ids.
```python
dataset.info
```

### Access Dataset Items
There are three methods to access individual Dataset Items:

(1) Dataset Items are accessible by reference id
```python
item = dataset.refloc("my_img_001.png")
```
(2) Dataset Items are accessible by index
```python
item = dataset.iloc(0)
```
(3) Dataset Items are accessible by the dataset_item_id assigned internally
```python
item = dataset.loc("dataset_item_id")
```

### Add Annotations
Upload groundtruth annotations for the items in your dataset.
Box2DAnnotation has same format as https://dashboard.scale.com/nucleus/docs/api#add-ground-truth
```python
response = dataset.annotate({"annotations:" [Box2DAnnotation, ..., Box2DAnnotation]})
```

For particularly large payloads, please reference the accompanying scripts in **references**

### Add Model
The model abstraction is intended to represent a unique architecture.
Models are independent of any dataset.

```python
response = client.add_model({"name": "My Model", "reference_id": "model-0.5", "metadata": {"iou_thr": 0.5}})
```

### Create Model Run
In contrast to the model abstraction, the model run abstraction
represents an experiment. A model run is associated with both a model and
a dataset.  A model run is meant to represent "the predictions of model y on
dataset x"

Creating a model run returns a ModelRun object.
```python
model_run = dataset.create_model_run({"reference_id": "model-0.5"})
```

### Get ModelRun Info
Returns the associated model_id, human-readable name of the run, status, and user specified metadata.
```python
model_run.info
```

### Upload Predictions to ModelRun
This method populates the model_run object with predictions.
Returns the associated model_id, human-readable name of the run, status, and user specified metadata.
Takes a list of Box2DPredictions within the payload, where Box2DPrediction
is formulated as in https://dashboard.scale.com/nucleus/docs/api#upload-model-outputs
```python
payload = {"annotations": List[Box2DPrediction]}
model_run.predict(payload)
```

### Accessing ModelRun Predictions
You can access the modelRun predictions for an individual dataset_item through three methods:

(1) user specified reference_id
```python
model_run.refloc("my_img_001.png")
```
(2) Index
```python
model_run.iloc(0)
```
(3) Internally maintained dataset_item_id
```python
model_run.loc("dataset_item_id")
```

### Commit ModelRun
The commit action indicates that the user is finished uploading predictions associated
with this model run.  Committing a model run kicks off Nucleus internal processes
to calculate performance metrics like IoU. After being committed, a ModelRun object becomes immutable.
```python
model_run.commit()
```

### Delete ModelRun
Delete a model run using the target model_run_id
A response code of 200 indicates successful deletion.
```python
client.delete_model_run("model_run_id")
```
