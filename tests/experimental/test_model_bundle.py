import pytest
import torch

from nucleus.experimental.hosted_inference_client import HostedInference
from nucleus.experimental.utils import _nucleus_ds_to_s3url_list
from tests.helpers import TEST_DATASET_ITEMS


@pytest.mark.integration
def test_add_model_bundle():
    # perhaps this belongs in some other test script? This is a pretty heavy/end-to-end test
    # Tests both client and server functionality
    # TODO does it make sense to use an actual user to make the requests?

    client = HostedInference(api_key="TODO")  # TODO set up nucleus pytest api key

    model_name = "TestModel"
    model = torch.nn.Linear(1, 1)  # probably should be something pytorch
    model.weight[0, 0] = 42
    model.bias[0] = 43

    def load_predict_fn(model):
        if torch.cuda.is_available():
            print("Using GPU for inference")
            device = torch.device("cuda")
            model.cuda()
        else:
            print("Using CPU for inference")
            device = torch.device("cpu")

        model.eval()

        def predict(preprocess_output):
            # 'model' refers to the model we got back from get_model()
            with torch.no_grad():
                model_output = model(preprocess_output.to(device))

            return model_output

        return predict

    model_bundle = client.add_model_bundle(
        model_bundle_name=model_name,
        model=model,
        load_predict_fn=load_predict_fn,
    )
    assert (
        model_bundle.name == model_name
    ), "Model bundle name is not correct"

    # TODO more granular tests?


def test_nucleus_ds_to_s3url_list(dataset):
    # This is a weird test, verification is roughly as hard as the actual code
    s3urls = _nucleus_ds_to_s3url_list(dataset)
    assert len(s3urls) == len(dataset.items), "number of s3urls isn't number of dataset.items"
    assert s3urls == [item.image_location for item in TEST_DATASET_ITEMS], "S3URLs are different from items' image_locations"
