import os
import time

from nucleus.experimental.hosted_inference_client import HostedInference
from nucleus.experimental.model_bundle import ModelBundle

# TODO Don't include this file in final pr
from nucleus.experimental.model_endpoint import ModelEndpoint


def create_dummy_bundle(hmi_client):
    def returns_returns_1(x):
        def returns_1(y):
            return 1

        return returns_1

    model = None
    load_predict_func = returns_returns_1
    hmi_client.add_model_bundle("return1", model, load_predict_func)


def create_endpoint(hmi_client):
    env_params = {
        "framework_type": "pytorch",
        "pytorch_version": "1.7.0",
        "cuda_version": "11.0",
        "cudnn_version": "cudnn8-devel",
    }

    mb = ModelBundle("abc123")

    # TODO out of date lol
    args = {
        "service_name": "seantest",
        "env_params": env_params,
        # "bundle_id": "abc123",
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

    model_endpoint = hmi_client.create_model_endpoint(**args)
    print(model_endpoint.endpoint_id)


def make_task_call(
    endpoint_name: str, dataset_id: str, upload_to_nucleus=True
):
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])

    dataset = client.get_dataset(dataset_id)

    model_endpoint = ModelEndpoint(
        endpoint_id=endpoint_name, client=hmi_client
    )

    model_endpoint_async_job = model_endpoint.create_run_job(dataset=dataset)

    while not model_endpoint_async_job.is_done(poll=True):
        print("Waiting for predictions to finish...")
        time.sleep(5)

    print("Predictions complete!")
    predictions = model_endpoint_async_job.get_responses()
    print(predictions)

    if upload_to_nucleus:
        ts = str(time.time())
        model_endpoint_async_job.upload_responses_to_nucleus(
            client,
            dataset,
            ts,
            model_name="Test HMI upload",
            model_ref_id=f"test_hmi_upload_{ts}",
        )


def temp_clone_pandaset():
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    public_dataset_client = nucleus.NucleusClient(
        os.environ["NUCLEUS_PUBLIC_DATASET_API_KEY"]
    )

    dataset = public_dataset_client.get_dataset(
        "ds_bwhjbyfb8mjj0ykagxf0"
    )  # Public Pandaset Dataset id
    dataset_items = dataset.items

    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])
    cloned_dataset = client.create_dataset("Pandaset clone - small")
    cloned_dataset_items = [
        nucleus.DatasetItem(
            image_location=x.image_location,
            reference_id=x.reference_id,
            metadata=x.metadata,
        )
        for x in dataset_items[:1]
    ]
    cloned_dataset.append(cloned_dataset_items)


# temp_clone_pandaset()

if __name__ == "__main__":
    # make_task_call(
    #     endpoint_name="yi-tf-test",
    #     #dataset_id="ds_c4wht080x81g060m0nfg",
    #     dataset_id="ds_c4x8s3m6n260060cngs0",
    #     upload_to_nucleus=True
    # )
    # temp_clone_pandaset()

    hmi_client = HostedInference(
        api_key="", endpoint="http://localhost:3000/v1/hosted_inference"
    )
    img_url = "s3://scale-ml-hosted-model-inference/tmp/hosted-model-inference-outputs/c3f3b5ed-f182-4fa1-bfa5-9b2e017feb74.pkl"
    # hmi_client.create_endpoint()
    # print(hmi_client.connection.post({}, "model_bundle_upload"))
    # create_dummy_bundle(hmi_client)
    env_params = {
        "framework_type": "pytorch",
        "pytorch_version": "1.7.0",
        "cuda_version": "11.0",
        "cudnn_version": "cudnn8-devel",
    }
    # me2 = hmi_client.create_model_endpoint(
    #     service_name="seantest2",
    #     env_params=env_params,
    #     model_bundle=ModelBundle(name="return1"),
    #     cpus=1,
    #     memory="4Gi",
    #     gpus=0,
    #     gpu_type="Doesn't Matter",
    #     min_workers=1,
    #     max_workers=1,
    #     per_worker=1,
    #     requirements=[],
    # )
    # print(me2)
    mes = hmi_client.get_model_endpoints()
    for me in mes:
        print(me)
    m1 = mes[0]
    print(hmi_client.sync_request("seantest2", img_url))
    async_task = hmi_client.async_request("seantest2", img_url)
    print(async_task)
    print(hmi_client.get_async_response(async_task))
    for i in range(10):
        time.sleep(1)
        print(hmi_client.get_async_response(async_task))
    # print(hmi_client.get_bundles())
