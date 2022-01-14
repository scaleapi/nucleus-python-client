# This file contains all the Core Nucleus <-> HMI integrations
import logging
from typing import Dict, List, Tuple

import cloudpickle
import smart_open
from boto3 import Session

import nucleus
from nucleus import Dataset, DatasetItem
from nucleus.dataset_item import DatasetItemType
from nucleus.deploy.model_endpoint import ModelEndpoint, ModelEndpointAsyncJob

logger = logging.getLogger(__name__)
logging.basicConfig()


class NucleusDatasetInferenceRun:
    """
    This class is temporary, long-term we want to move to Nucleus backend calling HMI directly
    """

    # TODO shoot HMI endpoints might not be authorized for nucleus s3urls ugh
    def __init__(
        self,
        hmi_async_job: ModelEndpointAsyncJob,
        nucleus_client,
        s3url_to_dataset_map,
        dataset,
    ):
        self.hmi_async_job = hmi_async_job
        self.nucleus_client = nucleus_client
        self.s3url_to_dataset_map = s3url_to_dataset_map
        self.dataset = dataset

    def is_done(self, poll=True):
        return self.hmi_async_job.is_done(poll=poll)

    def poll(self):
        return self.hmi_async_job.poll_endpoints()

    def upload_to_nucleus(
        self, model_run_name, model=None, model_name=None, model_ref_id=None
    ):
        """Upload model run responses to Nucleus.
        TODO we probably want to not have this function

        """
        # TODO untested
        # TODO we eventually want to move away from client side upload

        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")

        # Create a Nucleus model if we don't have one
        if model is None:
            assert (
                model_name is not None and model_ref_id is not None
            ), "If you don't pass a nucleus model you better pass a model name and reference id"
            model = self.nucleus_client.add_model(
                name=model_name, reference_id=model_ref_id
            )

        # Create a Nucleus model run
        model_run = model.create_run(
            name=model_run_name, dataset=self.dataset, predictions=[]
        )
        prediction_items = []
        for s3url, dataset_item in self.s3url_to_dataset_map.items():
            item_link = self.responses[s3url]
            print(f"item_link={item_link}")
            # e.g. s3://scale-ml/tmp/hosted-model-inference-outputs/a224499e-50ac-4b08-ad0c-c18e74c14184.pkl
            kwargs = {
                "transport_params": {
                    "session": Session(profile_name="ml-worker")
                }
            }

            with smart_open.open(item_link, "rb", **kwargs) as bundle_pkl:
                inference_result = cloudpickle.load(bundle_pkl)
                ref_id = dataset_item.reference_id
                for box in inference_result:
                    # TODO assuming box is a list of (x, y, w, h, label). This is almost certainly not the case.
                    # Also, label is probably returned as an integer instead of a label that makes semantic sense
                    pred_item = nucleus.BoxPrediction(
                        label=box["label"],
                        x=box["left"],
                        y=box["top"],
                        width=box["width"],
                        height=box["height"],
                        reference_id=ref_id,
                    )
                    prediction_items.append(pred_item)

        job = model_run.predict(prediction_items, asynchronous=True)
        job.sleep_until_complete()
        job.errors()


def create_nucleus_dataset_inference_run(
    hmi_endpoint: ModelEndpoint, nucleus_client, dataset: Dataset
):
    """
    Returns a NucleusDatasetInferenceRun, client will need to periodically call poll on this in order to upload
    """
    s3urls, s3url_to_dataset_map = _nucleus_ds_to_s3url_list(dataset)
    async_job = hmi_endpoint.predict(s3urls)
    return NucleusDatasetInferenceRun(
        async_job, nucleus_client, s3url_to_dataset_map, dataset
    )


def _nucleus_ds_to_s3url_list(
    dataset: Dataset,
) -> Tuple[List[str], Dict[str, DatasetItem]]:
    # TODO I'm not sure if dataset items are necessarily s3URLs. Does this matter?
    # TODO support lidar point clouds
    if len(dataset.items) == 0:
        logger.warning("Passed a dataset of length 0")
        return [], {}  # TODO return type?
    dataset_item_type = dataset.items[0].type
    if not all(data.type == dataset_item_type for data in dataset.items):
        logger.warning("Dataset has multiple item types")
        raise Exception  # TODO (code style) too broad exception

    s3url_to_dataset_map = {}
    # Do we need to keep track of nucleus ids?
    if dataset_item_type == DatasetItemType.IMAGE:
        s3Urls = [
            data.image_location
            for data in dataset.items
            if data.image_location is not None
        ]
        s3url_to_dataset_map = {
            data.image_location: data
            for data in dataset.items
            if data.image_location is not None
        }
    elif dataset_item_type == DatasetItemType.POINTCLOUD:
        s3Urls = [
            data.pointcloud_location
            for data in dataset.items
            if data.pointcloud_location is not None
        ]
        s3url_to_dataset_map = {
            data.pointcloud_location: data
            for data in dataset.items
            if data.pointcloud_location is not None
        }
    else:
        raise NotImplementedError(
            f"Dataset Item Type {dataset_item_type} not implemented"
        )
    # TODO for demo
    return (
        s3Urls,
        s3url_to_dataset_map,
    )  # TODO duplicated data in returned values
