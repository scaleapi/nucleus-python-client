# Changelog

All notable changes to the [Nucleus Python Client](https://github.com/scaleapi/nucleus-python-client) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.12.0) - 2022-05-27

### Added

- Allow users to create custom evaluation functions for Scenario Tests in Validate


## [0.11.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.11.2) - 2022-05-20

### Changed

- Restored backward compatibility of video constructor by adding back deprecated attachment_type argument

## [0.11.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.11.1) - 2022-05-19

### Added

- Exporting model predictions from a slice

## [0.11.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.11.0) - 2022-05-13

### Added

- Segmentation prediction masks can now be evaluated against polygon annotation with new Validate functions
- New function SegmentationToPolyIOU, configurable through client.validate.eval_functions.segmentation_to_poly_iou
- New function SegmentationToPolyRecall, configurable through client.validate.eval_functions.segmentation_to_poly_recall
- New function SegmentationToPolyPrecision, configurable through client.validate.eval_functions.segmentation_to_poly_precision
- New function SegmentationToPolyMAP, configurable through client.validate.eval_functions.segmentation_to_poly_map
- New function SegmentationToPolyAveragePrecision, configurable through client.validate.eval_functions.segmentation_to_poly_ap

## [0.10.8](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.8) - 2022-05-10

### Fixed

- Add checks for duplicate (`reference_id`, `annotation_id`) when uploading Annotations or Predictions

## [0.10.7](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.7) - 2022-05-09

### Fixed

- Add checks for duplicate reference IDs

## [0.10.6](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.6) - 2022-05-06

### Added

- Video privacy mode

### Changed

- Removed attachment_type argument in video upload API

## [0.10.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.5) - 2022-05-04

### Fixed

- Invalid polygons are dropped from PolygonMetric iou matching

## [0.10.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.4)) - 2022-05-02

### Added

- Additional check added for KeypointsAnnotation names validation
- MP4 video upload

## [0.10.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.3) - 2022-04-22

### Fixed

- Polygon and bounding box matching uses Shapely again providing faster evaluations
- Evaluation function passing fixed for Polygon and Boundingbox configurations

## [0.10.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.1) - 2022-04-21

### Added

- Added check for payload size

## [0.10.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.10.0)) - 2022-04-21

### Added

- `KeypointsAnnotation` added
- `KeypointsPrediction` added

## [0.9.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.9.0) - 2022-04-07

### Added

- Validate metrics support metadata and field filtering on input annotation and predictions
- 3D/Cuboid metrics: Recall, Precision, 3D IOU and birds eye 2D IOU```
- Shapely can be used for metric development if the optional scale-nucleus[shapely] is installed
- Full support for passing parameters to evaluation configurations

## [0.8.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.8.4) - 2022-04-06

- Changing `camera_params` of dataset items can now be done through the dataset method `update_items_metadata`

## [0.8.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.8.3) - 2022-03-29

### Added

- new Validate functionality to intialize scenario tests without a threshold, and to set test thresholds based on a baseline model.

## [0.8.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.8.2) - 2022-03-18

### Added

- a fix to the CameraModels enumeration to fix export of camera calibrations for 3D scenes

## [0.8.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.8.0) - 2022-03-18

### Added

- slice.items_generator() and dataset.items_generator() to allow for export of dataset items at any scale.

## [0.8.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.8.0) - 2022-03-16

### Added

- mask_url can now be a local file for segmentation annotations or predictions, meaning local upload is now supported for segmentations
- Camera params for sensor fusion ingest now support additional camera params to accommodate fisheye camera, etc.
- More detailed parameters to control for upload in case of timeouts (see dataset.upload_predictions, dataset.append, and dataset.upload_predictions)

### Fixed

- Artificially low concurrency for local uploads (all local uploads should be faster now)
- Client no longer uses the deprecated (and now removed) segmentation-specific server endpoints
- Fixed a bug where retries for local uploads were not working properly: should improve local upload robustness

### Removed

- client.predict, client.annotate, which have been marked as deprecated for several months.

## [0.7.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.7.0) - 2022-03-09

### Added

- `LineAnnotation` added
- `LinePrediction` added

## [0.6.7](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.7) - 2021-03-08

### Added

- `get_autotag_refinement_metrics`
- Get model using `model_run_id`
- Video API change to require `image_location` instead of `video_frame_location` in `DatasetItems`

## [0.6.6](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.6) - 2021-02-18

### Added

- Video upload support

## [0.6.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.5) - 2021-02-16

### Fixed

- `Dataset.update_autotag` docstring formatting
- `BoxPrediction` dataclass parameter typing
- `validate.scenario_test_evaluation` typo

## [0.6.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.4) - 2021-02-16

### Fixes

- Categorization metrics are patched to run properly on Validate evaluation service

## [0.6.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.3) - 2021-02-15

### Added

- Add categorization f1 score to metrics

## [0.6.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.1) - 2021-02-08

### Added

- Adapt scipy and click dependencies to allow Google COLAB usage without update

## [0.6.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.6.0) - 2021-02-07

### Added

- Nucleus CLI interface `nu`. Installation instructions are in the `README.md`.

## [0.5.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.5.4) - 2022-01-28

### Added

- Add `NucleusClient.get_job` to retrieve `AsyncJob`s by job ID

## [0.5.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.5.3) - 2022-01-25

### Added

- Add average precision to polygon metrics
- Add mean average precision to polygon metrics

## [0.5.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.5.2) - 2022-01-20

### Added

- Add `Dataset.delete_scene`

### Fixed

- Removed `Shapely` dependency

## [0.5.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.5.1) - 2022-01-11

### Fixed

- Updated dependencies for full Python 3.6 compatibility

## [0.5.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.5.0) - 2022-01-10

### Added

- `nucleus.metrics` module for computing metrics between Nucleus `Annotation` and `Prediction` objects.

## [0.4.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.5) - 2022-01-07

### Added

- `Dataset.scenes` property that fetches the Scale-generated ID, reference ID, type, and metadata of all scenes in the Dataset.

## [0.4.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.4.4) - 2022-01-04

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
