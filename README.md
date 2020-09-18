# Nucleus

## Installation

### Editable mode

`$ pip install -e . `

### As a Normal Package

`$ pip install git+ssh://git@github.com/scaleapi/nucleus-python-client.git`

## Usage

```buildoutcfg
import nucleus

client = nucleus.NucleusClient('YOUR_API_KEY_HERE')
```

### Create Dataset

```buildoutcfg
response = client.create_dataset({"name": "My Dataset"})
dataset = client.get_dataset(response["dataset_id"])
```

### Add Model

```buildoutcfg
response = client.add_model({"name": "My Model", "reference_id": "model-0.5", "metadata": {"iou_thr": 0.5}})
```

### Add Model Run

```buildoutcfg
dataset.create_model_run({"reference_id": "model-0.5"})
```
