import pytest
import torch

from nucleus.experimental.model_bundle import add_model_bundle


@pytest.mark.integration
def test_add_model_bundle():
    # perhaps this belongs in some other test script? This is a pretty heavy/end-to-end test
    # Tests both client and server functionality

    model_name = "TestModel"
    reference_id = "12345"
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

    model_bundle = add_model_bundle(model_name=model_name, model=model, load_predict_fn=load_predict_fn, reference_id=reference_id)
    assert model_bundle.name == f"{model_name}_{reference_id}", "Model bundle name is not correct"

    # TODO more granular tests?
