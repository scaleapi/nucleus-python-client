# Changelog

All notable changes to the [Nucleus Python Client](https://github.com/scaleapi/nucleus-python-client) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.17.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.17.0) - 2024-02-06

### Added
- Added `dataset.add_items_from_dir`
- Added pytest-xdist for test parallelization

### Fixes
- Fix test `test_models.test_remove_invalid_tag_from_model`


## [0.16.17](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.17) - 2024-01-29

### Fixes
- Update documentation

## [0.16.16](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.16) - 2024-01-25

### Fixes
- Minor fixes to docstring

## [0.16.15](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.15) - 2024-01-11

### Fixes
- Fix lidar concurrent lidar pointcloud to also return intensity in case it exists in the response. 

## [0.16.14](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.14) - 2024-01-03

### Fixes
- Open up Pydantic version requirements as was fixed in 0.16.11

## [0.16.13](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.13) - 2023-12-13

### Added
- Added `trained_slice_id` parameter to `dataset.upload_predictions()` to specify the slice ID used to train the model.

### Fixes
- Fix offset generation for image chips in `dataset.items_and_annotation_chip_generator()`

## [0.16.12](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.12) - 2023-11-29

### Added
- Added tag support for slices. 

Example:
```python
>>> slc = client.get_slice('slc_id')
>>> tags = slc.tags
>>> slc.add_tags(['new_tag_1', 'new_tag_2'])
```

## [0.16.11](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.11) - 2023-11-22

### Added
- Added `num_processes` parameter to `dataset.items_and_annotation_chip_generator()` to specify parallel processing.
- Method to allow for concurrent task fetches for pointcloud data

Example:
```python
>>> task_ids = ['task_1', 'task_2']
>>> resp = client.download_pointcloud_tasks(task_ids=task_ids, frame_num=1)
>>> resp
{
  'task_1': [Point3D(x=5, y=10.7, z=-2.3), ...],
  'task_2': [Point3D(x=1.3 y=11.1, z=1.5), ...],
}
```

### Fixes
- Support environments using pydantic>=2

## [0.16.10](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.10) - 2023-11-22

Allow creating a dataset by crawling all images in a directory, recursively. Also supports privacy mode datasets.

#### Example structure:
```
~/Documents/
    data/
        2022/
            - img01.png
            - img02.png
        2023/
            - img01.png
            - img02.png
```

#### Default Example:

```python
data_dir = "~/Documents/data"
client.create_dataset_from_dir(data_dir)
# this will create a dataset named "data" and will contain 4 images, with the ref IDs:
# ["2022/img01.png", "2022/img02.png", "2023/img01.png", "2023/img02.png"]
```

#### Example Privacy Mode:

This requires that a proxy (or file server) is setup  and can serve files _relative_ to the data_dir

```python
data_dir = "~/Documents/data"
client.create_dataset_from_dir(
    data_dir,
    dataset_name='my-dataset',
    use_privacy_mode=True,
    privacy_mode_proxy="http://localhost:5000/assets/"
)
```

This would create a dataset `my-dataset`, and when opened in Nucleus, the images would be requested to the path:
`<privacy_mode_proxy>/<img ref id>`, for example: `http://localhost:5000/assets/2022/img01.png`


## [0.16.9](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.9) - 2023-11-17

### Fixes

- Minor fixes to video scene upload on privacy mode

## [0.16.8](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.8) - 2023-11-16

### Added

#### Dataset Item width and height
- Allow passing width and height to `DatasetItem`
- This is _required_ when using privacy mode

#### Dataset Item Fetch
- Added `dataset.items_and_annotation_chip_generator()` functionality to generate chips of images in s3 or locally.
- Added `query` parameter for `dataset.items_and_annotation_generator()` to filter dataset items.

### Removed
- `upload_to_scale` is no longer a property in `DatasetItem`, users should instead specify `use_privacy_mode` on the dataset during creation


## [0.16.7](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.7) - 2023-11-03

### Added
- Allow direct embedding vector upload together with dataset items. `DatasetItem` now has an additional parameter called `embedding_info` which can be used to directly upload embeddings when a dataset is uploaded.
- Added `dataset.embedding_indexes` property, which exposes information about every embedding index which belongs to the dataset.   


## [0.16.6](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.6) - 2023-11-01

### Added
- Allow datasets to be created in "privacy mode". For example, `client.create_dataset('name', use_privacy_mode=True)`.
- Privacy Mode lets customers use Nucleus without sensitive raw data ever leaving their servers.
- When set to `True`, you can submit URLs to Nucleus that link to raw data assets like images or point clouds, instead of transferring that data to Scale. Access control is then completely in the hands of users: URLs may optionally be protected behind your corporate VPN or an IP whitelist. When you load a Nucleus web page, your browser will directly fetch the raw data from your servers without it ever being accessible to Scale.


## [0.16.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.5) - 2023-10-30

### Added
- Added a `description` to the slice info. 

### Changed
- Made `skeleton` key optional on `KeypointsAnnotation`. 


## [0.16.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.4) - 2023-10-23

### Added
- Added a `query_objects` method on the Dataset class.
- Example
```shell
>>> ds = client.get_dataset('ds_id')
>>> objects = ds.query_objects('annotations.metadata.distance_to_device > 150', ObjectQueryType.GROUND_TRUTH_ONLY)
[CuboidAnnotation(label="", dimensions={}, ...), ...]
```
- Added `EvaluationMatch` class to represent IOU Matches, False Positives and False Negatives retrieved through the `query_objects` method 


## [0.16.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.3) - 2023-10-10

### Added
- Added a `query_scenes` method on the Dataset class.
- Example
```shell
>>> ds = client.get_dataset('ds_id')
>>> scenes = ds.query_scenes('scene.metadata.foo = "baz"')
[Scene(reference_id="", metadata={}, ...), ...]
```


## [0.16.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.2) - 2023-10-03

### Fixed
- Raise error on all error states for AsyncJob.sleep_until_complete(). Before it only handled the deprecated "Errored"


## [0.16.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.1) - 2023-09-18

### Added
- Added `asynchronous` parameter for `slice.export_embeddings()` and `dataset.export_embeddings()` to allow embeddings to be exported asynchronously.

### Changed
- Changed `slice.export_embeddings()` and `dataset.export_embeddings()` to be asynchronous by deafult.

## [0.16.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.16.0) - 2023-09-18

### Removed
- Support for Python 3.6 - it is end of life for more than a year

### Fixed
- Development environment for Python 3.11
- 

## [0.15.11](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.11) - 2023-09-15

### Added
- Added `slice.export_raw_json()` functionality to support raw export of object slices (annotations, predictions, item and scene level data). Currently does not support image slices. 


## [0.15.10](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.10) - 2023-07-20

### Added
- Fix `slice.export_predictions(args)` and `slice.export_predictions_generator(args)` methods to return `Predictions` instead of `Annotations`

## [0.15.9](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.9) - 2023-06-26

### Added
- Support for Scale Launch client v1.0.0 and higher for the Nucleus + Launch integration

## [0.15.7](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.7) - 2023-06-09

### Added
- Allow for downloading pointcloud data for a give task and frame number, example:

```python
import nucleus
import numpy as np
client = nucleus.NucleusClient(API_KEY)
pts = client.download_pointcloud_task(task_id, frame_num=1)
np_pts = np.array([pt.to_list() for pt in pts])
```

## [0.15.6](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.6) - 2023-06-03

### Changed
- Document new restrictions to slice create/append.
- `Dataset.create_slice` and `Slice.append` methods cannot exceed 10,000 items per request.

## [0.15.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.5) - 2023-05-8

### Fixed
- Give default annotation_id to `KeypointAnnotations` when not specified


## [0.15.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.4) - 2023-03-21

### Changed
- Added `create_slice_by_ids` to create slices from dataset item, scene, and object IDs


## [0.15.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.3) - 2023-03-02

### Changed
- Allow denormalized scores in `EvaluationResult`s

## [0.15.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.2) - 2023-02-10

### Changed
- Fix `client.create_launch_model_from_dir(args)` method

## [0.15.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.1) - 2023-01-16

### Changed
- Better filter tuning of `client.list_jobs(args)` method

### Added
- Dataset method to filter jobs, and statistics on running jobs
Example:
```python
>>> client = nucleus.NucleusClient(API_KEY)
>>> ds = client.get_dataset(ds_id)
>>> ds.jobs(show_completed=True, stats_only=True)
{'autotagInference': {'Cancelled': 1, 'Completed': 11},
 'modelRunCommit': {'Completed': 7, 'Errored_Server': 1, 'Running': 1},
 'sliceQuery': {'Completed': 40, 'Running': 2}}
```

Detailed Example
```python
>>> from nucleus.job import CustomerJobTypes
>>> client = nucleus.NucleusClient(API_KEY)
>>> ds = client.get_dataset(ds_id)
>>> from_date = "2022-12-20"; to_date = "2023-01-15"
>>> job_types = [CustomerJobTypes.MODEL_INFERENCE_RUN, CustomerJobTypes.UPLOAD_DATASET_ITEMS]
>>> ds.jobs(
  from_date=from_date,
  to_date=to_date,
  show_completed=True,
  job_types=job_types,
  limit=150
)
# ... returns list of AsyncJob objects
```


## [0.15.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.15.0) - 2022-12-19

### Changed
- `dataset.slices` now returns a list of `Slice` objects instead of a list of IDs

### Added
Retrieve a slice from a dataset by its name, or all slices of a particular type from a dataset. Where type is one of `["dataset_item", "object", "scene"]`.
- `dataset.get_slices(name, slice_type): List[Slice]`
```python
from nucleus.slice import SliceType
dataset.get_slices(name="My Slice")
dataset.get_slices(slice_type=SliceType.DATASET_ITEM)
```

## [0.14.30](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.30) - 2022-11-29

### Added
- Support for uploading track-level metrics to external evaluation functions using track_ref_ids

## [0.14.29](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.29) - 2022-11-22

### Added
- Support for `Track`s, enabling ground truth annotations and model predictions to be grouped across dataset items and scenes
- Helpers to update track metadata, as well as to create and delete tracks at the dataset level


## [0.14.28](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.28) - 2022-11-17

### Added
- Support for appending to slice with scene reference IDs
- Better error handling when appending to a slice with non-existent reference IDs


## [0.14.27](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.27) - 2022-11-04

### Added
- Support for scene-level external evaluation functions
- Support for uploading custom scene-level metrics


## [0.14.26](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.26) - 2022-11-01

### Added
- Support for fetching scene from a `DatasetItem.reference_id`
Example:
```python
dataset = client.get_dataset("<dataset_id>")
assert dataset.is_scene  # only works on scene datasets
some_item = dataset.iloc(0)
dataset.get_scene_from_item_ref_id(some_item['item'].reference_id) 
```


## [0.14.25](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.25) - 2022-10-20

### Updated
- Items of a slice can be retrieved by Slice property `.item`
- The type of items returned from `.items` is based on the slice `type`:
  - `slice.type == 'dataset_item'` => list of `DatasetItem` objects
  - `slice.type == 'object'` => list of `Annotation`/`Prediction` objects
  - `slice.type == 'scene'` => list of `Scene` objects


## [0.14.24](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.24) - 2022-10-19

### Fixed
- Late imports for seldomly used heavy libraries. Sped up CLI invocation and autocomplation.
  If you had shell completions installed before we recommend removeing them from your .(bash|zsh)rc
  file and reinstalling with nu install-completions

## [0.14.23](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.23) - 2022-10-17

### Added
- Support for building slices via Nucleus' Smart Sample


## [0.14.22](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.22) - 2022-10-14

### Added
- Trigger for calculating Validate metrics for a model. This allows underperforming slice discovery and more model analysis


## [0.14.21](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.21) - 2022-09-28

### Added
- Support for `context_attachment` metadata values. See [upload metadata](https://nucleus.scale.com/docs/upload-metadata) for more information.


## [0.14.20](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.20) - 2022-09-23

### Fixed
- Local uploads are correctly batched and prevents flooding the network with requests

## [0.14.19](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.19) - 2022-08-26

### Added
- Support for Coordinate metadata values. See [upload metadata](https://nucleus.scale.com/docs/upload-metadata) for more information.

## [0.14.18](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.18) - 2022-08-16

### Added
- Metadata and confidence support for scene categories

## [0.14.17](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.17) - 2022-08-15

### Fixed
- Fix `AsyncJob` status payload keys causing test failures
- Fix `AsyncJob` export test
- Fix `page_size` for `{Dataset,Slice}.items_and_annotatation_generator()`
- Change to simple dependency install step to fix CircleCI caching failures

## [0.14.16](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.16) - 2022-08-12

### Added
- Scene categorization support

## [0.14.15](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.15) - 2022-08-11

### Removed
- Removed s3fs, fsspec dependencies for simpler installation in various environments

## [0.14.14](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.14) - 2022-08-11

### Added
- client.slices to list all of users slices independent of dataset
- Added optional parameter `asynchronous: bool` to `Dataset.update_item_metadata` and  `Dataset.update_scene_metadata`,
allowing the update to run as a background job when set to `True`

### Fixed
- Validate unit test listing and evaluation history listing. Now uses new bulk fetch endpoints for faster listing.


## [0.14.13](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.13) - 2022-08-10

### Fixed
- Fix payload parsing for scene export


## [0.14.12](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.12) - 2022-08-05

### Added
- Added auto-paginated `Slice.export_predictions_generator`

### Fixed
- Change `{Dataset,Slice}.items_and_annotation_generator` to work with improved paginate endpoint


## [0.14.11](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.11) - 2022-07-20

### Fixed
- Various docstring and typing updates

## [0.14.10](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.10) - 2022-07-20

### Added
- `Dataset.items_and_annotation_generator()`

### Fixed
- `Slice.items_and_annotation_generator()` bug

## [0.14.9](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.9) - 2022-07-14

### Fixed
- NoneType errors in Validate

## [0.14.8](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.8) - 2022-07-14

### Fixed
- Segmentation metrics filtering. Prior version artificially boosted performance when filtering was applied.

## [0.14.7](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.7) - 2022-07-07

### Added
- Support running structured queries and retrieving item results via API

## [0.14.6](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.6) - 2022-07-07

### Fixed
- `Dataset.delete_annotations` now defaults `reference_ids` to an empty list and `keep_history` to true

## [0.14.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.5) - 2022-07-05

### Fixed
- Averaging of rich semantic segmentation taxonomies not taking into account missing classes

## [0.14.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.4) - 2022-06-21

### Fixed
- Regression that caused Validate filter statements to not work

## [0.14.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.3) - 2022-06-21

### Fixed
- CLI installation without GEOS errored out. Now handled by importer.


## [0.14.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.2) - 2022-06-21

### Fixed
- Better error reporting when everything is filtered out by a filter statement in a Validate evaluation function

## [0.14.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.1) - 2022-06-20

### Fixed
- Adapt Segmentation metrics to better support instance segmentation
- Change Segmentation/Polygon metrics to use new segmentation metrics 


## [0.14.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.14.0) - 2022-06-16

### Added

- Allow creation/deletion of model tags on new and existing models, eg:
```python
# on model creation
model = client.create_model(name="foo_model", reference_id="foo-model-ref", tags=["some tag"])

# on existing models
existing_model = client.models[0]
existing_model.add_tags(['tag a', 'tag b'])

# remove tag
existing_model.remove_tags(['tag a'])
```

## [0.13.5](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.13.4) - 2022-06-15

### Fixed
- Guard against invalid skeleton indexes in KeypointsAnnotation


## [0.13.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.13.4) - 2022-06-09

### Fixed
- Guard against extras imports 

## [0.13.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.13.3) - 2022-06-09

### Fixed
- Make installation of scale-launch optional (again!).

## [0.13.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.13.2) - 2022-06-08

### Fixed

- Open up requirements for easier installation in more environments. Add more optional installs under `metrics`
 
## [0.13.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.13.1) - 2022-06-08

### Fixed

- Make installation of scale-launch optional

## [0.13.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.13.0) - 2022-06-08

### Added

- Segmentation functions to Validate API

## [0.12.4](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.12.4) - 2022-06-02

### Fixed

- Poetry dependency list


## [0.12.3](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.12.3) - 2022-06-02

### Added

- New methods to export associated Scale task info at either the item or scene level.
- `Dataset.export_scale_task_info`
- `Slice.export_scale_task_info`


## [0.12.2](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.12.2) - 2022-06-02

### Added

- Allow users to upload external evaluation results calculated on the client side.


## [0.12.1](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.12.1) - 2022-06-02

### Added

- Suppress warning statement when un-implemented standard configs found

## [0.12.0](https://github.com/scaleapi/nucleus-python-client/releases/tag/v0.12.0) - 2022-05-27

### Added

- Allow users to create external evaluation functions for Scenario Tests in Validate.


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
