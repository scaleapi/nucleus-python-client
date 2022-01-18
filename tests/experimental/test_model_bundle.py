import time

import pytest
import torch

from nucleus.deploy.nucleus_integration import _nucleus_ds_to_s3url_list


@pytest.mark.integration
def test_add_model_bundle(DEPLOY_CLIENT):
    # perhaps this belongs in some other test script? This is a pretty heavy/end-to-end test
    # Tests both client and server functionality
    # TODO does it make sense to use an actual user to make the requests?

    client = DEPLOY_CLIENT  # TODO set up nucleus pytest api key

    model_name = f"TestModel_{int(time.time())}"  # Unfortunately we don't have a way of deleting this from the database
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

    model_bundle = client.create_model_bundle(
        model_bundle_name=model_name,
        model=model,
        load_predict_fn=load_predict_fn,
    )
    assert model_bundle.name == model_name, "Model bundle name is not correct"

    # TODO more granular tests?


def test_nucleus_ds_to_s3url_list(dataset):
    # This is a weird test, verification is roughly as hard as the actual code
    s3urls, _ = _nucleus_ds_to_s3url_list(dataset)
    assert len(s3urls) == len(
        dataset.items
    ), "number of s3urls isn't number of dataset.items"
    assert s3urls == [
        item.image_location for item in dataset.items
    ], "S3URLs are different from items' image_locations"
