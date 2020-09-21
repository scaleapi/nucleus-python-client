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

### Add Model

```python
response = client.add_model({"name": "My Model", "reference_id": "model-0.5", "metadata": {"iou_thr": 0.5}})
```

### Add Model Run

```python
dataset.create_model_run({"reference_id": "model-0.5"})
```
