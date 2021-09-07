import logging
import time
import os

import nucleus
from nucleus.experimental.model_bundle import *
# TODO Don't include this file in final pr


def create_endpoint():
    env_params = {
        "framework_type": "pytorch",
        "pytorch_version": "1.7.0",
        "cuda_version": "11.0",
        "cudnn_version": "cudnn8-devel",
    }

    mb = ModelBundle("abc123")

    args = {
        "service_name": "seantest",
        "env_params": env_params,
        #"bundle_id": "abc123",
        "model_bundle": mb,
        "cpus": 1,
        "memory": "4Gi",
        "gpus": 1,
        "gpu_type": "nvidia-tesla-t4",
        "min_workers": 1,
        "max_workers": 1,
        "per_worker": 1,
        "requirements": [],
    }

    model_endpoint = create_model_endpoint(**args)
    print(model_endpoint.endpoint_url, model_endpoint.endpoint_name)


def make_task_call(upload_to_nucleus = True):
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])
    datasets = client.list_datasets()
    print(datasets)

    #ds = client.get_dataset(datasets['dataset_ids'][0])  # Hopefully this is a small dataset
    ds = client.get_dataset("ds_c4s8stfv54200487yvrg")  # pandaset clone in sean's account

    model_endpoint = ModelEndpoint(endpoint_name="seantest", endpoint_url="idk")
    model_endpoint_async_job = model_endpoint.create_run_job(dataset=ds)

    while not model_endpoint_async_job.is_done(poll=True):
        print("Waiting for predictions to finish...")
        time.sleep(5)

    print("Predictions complete!")
    print(model_endpoint_async_job.get_responses())

    if upload_to_nucleus:
        model_endpoint_async_job.upload_responses_to_nucleus(client, ds, str(time.time()), model_name="Test HMI upload", model_ref_id="test_hmi_upload")

urls = [
    "S3://scaleapi-cust-lidar/Hesai/raw_data/2019-5-11/hesai_data_1557539856/undistorted/back_camera/1557539884.400195lf.jpg",
    "S3://scaleapi-cust-lidar/Hesai/raw_data/2019-5-11/hesai_data_1557539856/undistorted/back_camera/1557539885.999947lf.jpg",
    "S3://scaleapi-cust-lidar/Hesai/raw_data/2019-5-11/hesai_data_1557539856/undistorted/back_camera/1557539887.600534lf.jpg",
    "S3://scaleapi-cust-lidar/Hesai/raw_data/2019-5-11/hesai_data_1557539856/undistorted/back_camera/1557539889.200816lf.jpg",
    "S3://scaleapi-cust-lidar/Hesai/raw_data/2019-5-11/hesai_data_1557539856/undistorted/back_camera/1557539890.800669lf.jpg",
]


def temp_clone_pandaset():
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])
    #pandaset = client.get_dataset("ds_bwhjbyfb8mjj0ykagxf0")
    #print(pandaset)
    new_dataset = client.create_dataset("Pandaset clone")
    # TODO clone some dataset_items over into the new dataset
    new_dis = []
    #ditems = pandaset.items
    for i in range(5):
        new_di = nucleus.DatasetItem(image_location=urls[i], reference_id=str(i))
        new_dis.append(new_di)

    print(new_dis)
    new_dataset.append(new_dis)


# create_endpoint()
make_task_call()

#temp_clone_pandaset()
