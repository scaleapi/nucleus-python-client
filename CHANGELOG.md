# Changelog
All notable changes to the [Nucleus Python Client](https://github.com/scaleapi/nucleus-python-client) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.3.0) - 2021-11-23

### Added
- `NucleusClient.datasets` property that lists Datasets in a human friendlier manner than `NucleusClient.list_datasets()`
- `NucleusClient.models` property, this is preferred over the deprecated `list_models`
- `NucleusClient.jobs` property. `NucleusClient.list_jobs` is still the preferred method to use if you filter jobs on access.
- Deprecated method access now produces a deprecation warning in the logs.

### Deprecated
- `NucleusClient.list_datasets()` is deprecated in favor of `NucleusClient.datasets`. The latter allows for direct usage of `Dataset` objects.
- `NucleusClient.list_models()` is deprecated in favor of `NucleusClient.models`.
- `NucleusClient.get_dataset_items` is deprecated in favor of `Dataset.items` to make the object model more consistent.
- `NucleusClient.delete_dataset_item` is deprecated in favor of `Dataset.delete_item` to make the object model more consistent.
- `NucleusClient.populate_dataset` is deprecated in favor of `Dataset.append` to make the object model more consistent.
- `NucleusClient.ingest_tasks` is deprecated in favor of `Dataset.ingest_tasks` to make the object model more consistent.
- `NucleusClient.add_model` is deprecated in favor of `NucleusClient.create_model` for consistent terminology.
- `NucleusClient.dataset_info` is deprecated in favor of `Dataset.info` to make the object model more consistent.
- `NucleusClient.delete_annotations` is deprecated in favor of `Dataset.delete_annotations` to make the object model more consistent.

## [0.0.1 - 0.2.1](https://github.com/scaleapi/nucleus-python-client/releases)

[Refer to GitHub release notes for older releases.](https://github.com/scaleapi/nucleus-python-client/releases)
