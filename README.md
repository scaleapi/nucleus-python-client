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

**Best practices for testing:**
(1). Please run pytest from the root directory of the repo, i.e.

```
poetry run pytest tests/test_dataset.py
```

(2) To skip slow integration tests that have to wait for an async job to start.

```
poetry run pytest -m "not integration"
```
