# Nucleus

https://dashboard.scale.com/nucleus

Aggregate metrics in ML are not good enough. To improve production ML, you need to understand their qualitative failure modes, fix them by gathering more data, and curate diverse scenarios.

Scale Nucleus helps you:

* Visualize your data
* Curate interesting slices within your dataset
* Review and manage annotations
* Measure and debug your model performance

Nucleus is a new way—the right way—to develop ML models, helping us move away from the concept of one dataset and towards a paradigm of collections of scenarios.



## Installation

`$ pip install scale-nucleus`

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
dataset = client.create_dataset("My Dataset")
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
You can append both local images and images from the web. Simply specify the location and Nucleus will automatically infer if it's remote or a local file.
```python
dataset_item_1 = DatasetItem(image_location="./1.jpeg", reference_id="1", metadata={"key": "value"})
dataset_item_2 = DatasetItem(image_location="s3://srikanth-nucleus/9-1.jpg", reference_id="2", metadata={"key": "value"})
```

The append function expects a list of `DatasetItem` objects to upload, like this:
```python
response = dataset.append([dataset_item_1, dataset_item_2])
```

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
annotation_1 = BoxAnnotation(reference_id="1", label="label", x=0, y=0, width=10, height=10, annotation_id="ann_1", metadata={})
annotation_2 = BoxAnnotation(reference_id="2", label="label", x=0, y=0, width=10, height=10, annotation_id="ann_2", metadata={})
response = dataset.annotate([annotation_1, annotation_2])
```

For particularly large payloads, please reference the accompanying scripts in **references**

### Add Model
The model abstraction is intended to represent a unique architecture.
Models are independent of any dataset.

```python
model = client.add_model(name="My Model", reference_id="newest-cnn-its-new", metadata={"timestamp": "121012401"})
```

### Upload Predictions to ModelRun
This method populates the model_run object with predictions. `ModelRun` objects need to reference a `Dataset` that has been created.
Returns the associated model_id, human-readable name of the run, status, and user specified metadata.
Takes a list of Box2DPredictions within the payload, where Box2DPrediction
is formulated as in https://dashboard.scale.com/nucleus/docs/api#upload-model-outputs
```python
prediction_1 = BoxPrediction(reference_id="1", label="label", x=0, y=0, width=10, height=10, annotation_id="pred_1", confidence=0.9)
prediction_2 = BoxPrediction(reference_id="2", label="label", x=0, y=0, width=10, height=10, annotation_id="pred_2", confidence=0.2)

model_run = model.create_run(name="My Model Run", metadata={"timestamp": "121012401"}, dataset=dataset, predictions=[prediction_1, prediction_2])
```

### Commit ModelRun
The commit action indicates that the user is finished uploading predictions associated
with this model run.  Committing a model run kicks off Nucleus internal processes
to calculate performance metrics like IoU. After being committed, a ModelRun object becomes immutable.
```python
model_run.commit()
```

### Get ModelRun Info
Returns the associated model_id, human-readable name of the run, status, and user specified metadata.
```python
model_run.info
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

### Delete ModelRun
Delete a model run using the target model_run_id.

A response code of 200 indicates successful deletion.
```python
client.delete_model_run("model_run_id")
```

## For Developers

Clone from github and install as editable

```
git clone git@github.com:scaleapi/nucleus-python-client.git
cd nucleus-python-client
pip3 install poetry
poetry install
```

Please install the pre-commit hooks by running the following command:
```python
poetry run pre-commit install
```

**Best practices for testing:**
(1). Please run pytest from the root directory of the repo, i.e.
```
poetry pytest tests/test_dataset.py
```


