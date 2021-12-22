from typing import Sequence
import logging

from nucleus import Dataset
from nucleus.dataset_item import DatasetItemType

logger = logging.getLogger(__name__)
logging.basicConfig()


def _nucleus_ds_to_s3url_list(dataset: Dataset) -> Sequence[str]:
    # TODO I'm not sure if dataset items are necessarily s3URLs. Does this matter?
    # TODO support lidar point clouds
    if len(dataset.items) == 0:
        logger.warning("Passed a dataset of length 0")
        return None  # TODO return type?
    dataset_item_type = dataset.items[0].type
    if not all([data.type == dataset_item_type for data in dataset.items]):
        logger.warning("Dataset has multiple item types")
        raise Exception  # TODO (code style) too broad exception

    s3url_to_dataset_map = {}
    # Do we need to keep track of nucleus ids?
    if dataset_item_type == DatasetItemType.IMAGE:
        s3Urls = [data.image_location for data in dataset.items]
        s3url_to_dataset_map = {data.image_location: data for data in dataset.items}
    elif dataset_item_type == DatasetItemType.POINTCLOUD:
        s3Urls = [data.pointcloud_location for data in dataset.items]
        s3url_to_dataset_map = {data.pointcloud_location: data for data in dataset.items}
    else:
        raise NotImplementedError(f"Dataset Item Type {dataset_item_type} not implemented")
    # TODO for demo
    return s3Urls, s3url_to_dataset_map  # TODO duplicated data in returned values
