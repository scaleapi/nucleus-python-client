# Changelog
All notable changes to the [Nucleus Python Client](https://github.com/scaleapi/nucleus-python-client) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.4) - 2021-01-07

### Added
- `Dataset.scenes` property that fetches the Scale-generated ID, reference ID, type, and metadata of all scenes in the Dataset.

## [0.4.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.4) - 2021-01-04

### Added
- `Slice.export_raw_items()` method that fetches accessible (signed) URLs for all items in the Slice.

## [0.4.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.3) - 2022-01-03

### Added
- Improved error messages for categorization

### Changed
- Category taxonomies are now updatable

## [0.4.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.2) - 2021-12-16

### Added
- `Slice.name` property that fetches the Slice's user-defined name.
  - The Slice's items are no longer fetched unnecessarily; this used to cause considerable latency.
- `Slice.items` property that fetches all items contained in the Slice.

### Changed
- `Slice.info()` now only retrieves the Slice's `name`, `slice_id`, and `dataset_id`.
  - The Slice's items are no longer fetched unnecessarily; this used to cause considerable latency.
  - This method issues a warning to use `Slice.items` when attempting to `items`.

[###](###) Deprecated
- `NucleusClient.slice_info(..)` is deprecated in favor of `Slice.info()`.

## [0.4.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.1) - 2021-12-13

### Changed
- Datasets in Nucleus now fall under two categories: scene or item.
  - Scene Datasets can only have scenes uploaded to them.
  - Item Datasets can only have items uploaded to them.
- `NucleusClient.create_dataset` now requires a boolean parameter `is_scene` to immutably set whether the Dataset is a scene or item Dataset.

## [0.4.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.0) - 2021-08-12

### Added
- `NucleusClient.modelci` client extension that houses all features related to Model CI, a continuous integration and testing framework for evaluation machine learning models.
- `NucleusClient.modelci.UnitTest`- class to represent a Model CI unit test.
- `NucleusClient.modelci.UnitTestEvaluation`- class to represent an evaluation result of a Model CI unit test.
- `NucleusClient.modelci.UnitTestItemEvaluation`- class to represent an evaluation result of an individual dataset item within a Model CI unit test.
- `NucleusClient.modelci.eval_functions`- Collection class housing a library of standard evaluation functions used in computer vision.

## [0.3.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.3.0) - 2021-11-23

### Added
- `NucleusClient.datasets` property that lists Datasets in a human friendlier manner than `NucleusClient.list_datasets()`
- `NucleusClient.models` property, this is preferred over the deprecated `list_models`
- `NucleusClient.jobs` property. `NucleusClient.list_jobs` is still the preferred method to use if you filter jobs on access.
- Deprecated method access now produces a deprecation warning in the logs.

### Deprecated
- Model runs have been deprecated and will be removed in the near future. Use a Model directly instead. The following
  functions have all been deprecated as a part of that.
  - `NucleusClient.get_model_run(..)`
  - `NucleusClient.delete_model_run(..)`
  - `NucleusClient.create_model_run(..)`
  - `NucleusClient.commit_model_run(..)`
  - `NucleusClient.model_run_info(..)`
  - `NucleusClient.predictions_ref_id(..)`
  - `NucleusClient.predictions_iloc(..)`
  - `NucleusClient.predictions_loc(..)`
  - `Dataset.create_model_run(..)`
  - `Dataset.model_runs(..)`
- `NucleusClient.list_datasets` is deprecated in favor of `NucleusClient.datasets`. The latter allows for direct usage of `Dataset` objects.
- `NucleusClient.list_models` is deprecated in favor of `NucleusClient.models`.
- `NucleusClient.get_dataset_items` is deprecated in favor of `Dataset.items` to make the object model more consistent.
- `NucleusClient.delete_dataset_item` is deprecated in favor of `Dataset.delete_item` to make the object model more consistent.
- `NucleusClient.populate_dataset` is deprecated in favor of `Dataset.append` to make the object model more consistent.
- `NucleusClient.ingest_tasks` is deprecated in favor of `Dataset.ingest_tasks` to make the object model more consistent.
- `NucleusClient.add_model` is deprecated in favor of `NucleusClient.create_model` for consistent terminology.
- `NucleusClient.dataset_info` is deprecated in favor of `Dataset.info` to make the object model more consistent.
- `NucleusClient.delete_annotations` is deprecated in favor of `Dataset.delete_annotations` to make the object model more consistent.
- `NucleusClient.predict` is deprecated in favor of `Dataset.upload_predictions` to make the object model more consistent.
- `NucleusClient.dataitem_ref_id` is deprecated in favor of `Dataset.refloc` to make the object model more consistent.
- `NucleusClient.dataitem_iloc` is deprecated in favor of `Dataset.iloc` to make the object model more consistent.
- `NucleusClient.dataitem_loc` is deprecated in favor of `Dataset.loc` to make the object model more consistent.
- `NucleusClient.create_slice` is deprecated in favor of `Dataset.create_slice` to make the object model more consistent.
- `NucleusClient.create_custom_index` is deprecated in favor of `Dataset.create_custom_index` to make the object model more consistent.
- `NucleusClient.delete_custom_index` is deprecated in favor of `Dataset.delete_custom_index` to make the object model more consistent.
- `NucleusClient.set_continuous_indexing` is deprecated in favor of `Dataset.set_continuous_indexing` to make the object model more consistent.
- `NucleusClient.create_image_index` is deprecated in favor of `Dataset.create_image_index` to make the object model more consistent.
- `NucleusClient.create_object_index` is deprecated in favor of `Dataset.create_object_index` to make the object model more consistent.
- `Dataset.append_scenes` is deprecated in favor of `Dataset.append` for a simpler interface.

## [0.0.1 - 0.2.1](https://github.com/scaleapi/nucleus-python-client/releases)

[Refer to GitHub release notes for older releases.](https://github.com/scaleapi/nucleus-python-client/releases)
