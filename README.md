# Nucleus

https://dashboard.scale.com/nucleus

Aggregate metrics in ML are not good enough. To improve production ML, you need to understand their qualitative failure modes, fix them by gathering more data, and curate diverse scenarios.

Scale Nucleus helps you:

- Visualize your data
- Curate interesting slices within your dataset
- Review and manage annotations
- Measure and debug your model performance

Nucleus is a new way—the right way—to develop ML models, helping us move away from the concept of one dataset and towards a paradigm of collections of scenarios.

## Installation

`$ pip install scale-nucleus`

## CLI installation

We recommend installing the CLI via `pipx` (https://pypa.github.io/pipx/installation/). This makes sure that
the CLI does not interfere with you system packages and is accessible from your favorite terminal.

For MacOS:

```bash
brew install pipx
pipx ensurepath
pipx install scale-nucleus
# Optional installation of shell completion (for bash, zsh or fish)
nu install-completions
```

Otherwise, install via pip (requires pip 19.0 or later):

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
python3 -m pipx install scale-nucleus
# Optional installation of shell completion (for bash, zsh or fish)
nu install-completions
```

## Common issues/FAQ

### Outdated Client

Nucleus is iterating rapidly and as a result we do not always perfectly preserve backwards compatibility with older versions of the client. If you run into any unexpected error, it's a good idea to upgrade your version of the client by running

```
pip install --upgrade scale-nucleus
```

## Usage

For the most up to date documentation, reference: https://dashboard.scale.com/nucleus/docs/api?language=python.

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

When releasing a new version please add release notes to the changelog in `CHANGELOG.md`.

**Best practices for testing:**
(1). Please run pytest from the root directory of the repo, i.e.

```
poetry run pytest tests/test_dataset.py
```

(2) To skip slow integration tests that have to wait for an async job to start.

```
poetry run pytest -m "not integration"
```

## Pydantic Models

Prefer using [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/) models rather than creating raw dictionaries
or dataclasses to send or receive over the wire as JSONs. Pydantic is created with data validation in mind and provides very clear error
messages when it encounters a problem with the payload.

The Pydantic model(s) should mirror the payload to send. To represent a JSON payload that looks like this:

```json
{
  "example_json_with_info": {
    "metadata": {
      "frame": 0
    },
    "reference_id": "frame0",
    "url": "s3://example/scale_nucleus/2021/lidar/0038711321865000.json",
    "type": "pointcloud"
  },
  "example_image_with_info": {
    "metadata": {
      "author": "Picasso"
    },
    "reference_id": "frame0",
    "url": "s3://bucket/0038711321865000.jpg",
    "type": "image"
  }
}
```

Could be represented as the following structure. Note that the field names map to the JSON keys and the usage of field
validators (`@validator`).

```python
import os.path
from pydantic import BaseModel, validator
from typing import Literal


class JsonWithInfo(BaseModel):
    metadata: dict  # any dict is valid
    reference_id: str
    url: str
    type: Literal["pointcloud", "recipe"]

    @validator("url")
    def has_json_extension(cls, v):
        if not v.endswith(".json"):
            raise ValueError(f"Expected '.json' extension got {v}")
        return v


class ImageWithInfo(BaseModel):
    metadata: dict  # any dict is valid
    reference_id: str
    url: str
    type: Literal["image", "mask"]

    @validator("url")
    def has_valid_extension(cls, v):
        valid_extensions = {".jpg", ".jpeg", ".png", ".tiff"}
        _, extension = os.path.splitext(v)
        if extension not in valid_extensions:
            raise ValueError(f"Expected extension in {valid_extensions} got {v}")
        return v


class ExampleNestedModel(BaseModel):
    example_json_with_info: JsonWithInfo
    example_image_with_info: ImageWithInfo

# Usage:
import requests
payload = requests.get("/example")
parsed_model = ExampleNestedModel.parse_obj(payload.json())
requests.post("example/post_to", json=parsed_model.dict())
```

### Migrating to Pydantic

- When migrating an interface from a dictionary use `nucleus.pydantic_base.DictCompatibleModel`. That allows you to get
  the benefits of Pydantic but maintaints backwards compatibility with a Python dictionary by delegating `__getitem__` to
  fields.
- When migrating a frozen dataclass use `nucleus.pydantic_base.ImmutableModel`. That is a base class set up to be
  immutable after initialization.

**Updating documentation:**
We use [Sphinx](https://www.sphinx-doc.org/en/master/) to autogenerate our API Reference from docstrings.

To test your local docstring changes, run the following commands from the repository's root directory:

```
poetry shell
cd docs
sphinx-autobuild . ./_build/html --watch ../nucleus
```

`sphinx-autobuild` will spin up a server on localhost (port 8000 by default) that will watch for and automatically rebuild a version of the API reference based on your local docstring changes.

## Custom Metrics using Shapely in scale-validate

Certain metrics use `shapely` which is added as an optional dependency.

```bash
pip install scale-nucleus[metrics]
```

Note that you might need to install a local GEOS package since Shapely doesn't provide binaries bundled with GEOS for every platform.

```bash
#Mac OS
brew install geos
# Ubuntu/Debian flavors
apt-get install libgeos-dev
```

To develop it locally use

`poetry install --extras shapely`
