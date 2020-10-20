import tqdm
import tqdm.notebook
import nucleus
from nucleus.upload_response import UploadResponse
from nucleus.constants import (
    ITEMS_KEY,
    ANNOTATIONS_KEY,
    ANNOTATIONS_PROCESSED_KEY,
    PREDICTIONS_PROCESSED_KEY,
    STATUS_CODE_KEY,
    ERRORS_KEY,
    DATASET_ID_KEY,
    MODEL_RUN_ID_KEY,
)


def batch_upload_append(
    api_key, dataset_id, payload, batch_size=1000, use_notebook=False
):
    """
    Takes large payload, splits it into batches, and calls append sequentially.
    :param api_key: Scale API_KEY used to authenticate user.
    :param dataset_id: unique identifier of Nucleus dataset to which the items in payload will be uploaded.
    :param payload: json object representing payload to append operation.  Format should be consistent with
    dataset.append() input documented in nucleus folder.
    :param batch_size: (optional) number of payload items to be uploaded per batch, defaults to 1000.
    :param use_notebook: toggles between normal tqdm bar and tqdm bar for ipynb notebook.  User should specify
    use_notebook=True when invoking this function from a notebook environment.
    """
    client = nucleus.NucleusClient(api_key)
    items = payload[ITEMS_KEY]
    dataset = client.get_dataset(dataset_id)

    print("uploading {} items...".format(len(items)))
    batches = [
        items[i : i + batch_size] for i in range(0, len(items), batch_size)
    ]
    agg_response = UploadResponse(json={DATASET_ID_KEY: dataset_id})
    tqdm_batches = (
        tqdm.notebook.tqdm(batches) if use_notebook else tqdm.tqdm(batches)
    )
    for batch in tqdm_batches:
        response = dataset.append({ITEMS_KEY: batch})
        agg_response.update_response(response.json())
    print("upload complete!")
    return agg_response.json()


def batch_upload_annotation(
    api_key, dataset_id, payload, batch_size=1000, use_notebook=False
):
    """
    Takes large annotation payload, splits it into batches, and calls append sequentially.
    :param api_key: Scale API_KEY used to authenticate user.
    :param dataset_id: unique identifier of Nucleus dataset to which the annotations in payload will be uploaded.
    :param payload: json object representing payload to append operation.  Format should be consistent with
    dataset.annotate() input documented in nucleus folder.
    :param batch_size: (optional) number of payload items to be uploaded per batch, defaults to 1000.
    :param use_notebook: toggles between normal tqdm bar and tqdm bar for ipynb notebook.  User should specify
    use_notebook=True when invoking this function from a notebook environment.
    """
    client = nucleus.NucleusClient(api_key)
    items = payload[ANNOTATIONS_KEY]
    print("uploading {} items...".format(len(items)))
    dataset = client.get_dataset(dataset_id)
    batches = [
        items[i : i + batch_size] for i in range(0, len(items), batch_size)
    ]
    agg_response = {DATASET_ID_KEY: dataset_id, ANNOTATIONS_PROCESSED_KEY: 0}
    tqdm_batches = (
        tqdm.notebook.tqdm(batches) if use_notebook else tqdm.tqdm(batches)
    )
    for batch in tqdm_batches:
        response = dataset.annotate({ANNOTATIONS_KEY: batch})
        if STATUS_CODE_KEY in response:
            agg_response[ERRORS_KEY] = response
        else:
            agg_response[ANNOTATIONS_PROCESSED_KEY] += response[
                ANNOTATIONS_PROCESSED_KEY
            ]
    print("upload complete!")
    return agg_response


def batch_upload_prediction(
    api_key,
    dataset_id,
    model_run_id,
    payload,
    batch_size=1000,
    use_notebook=False,
):
    """
    Takes large model prediction, splits it into batches, and calls append sequentially.
    :param api_key: Scale API_KEY used to authenticate user.
    :param dataset_id: unique identifier of Nucleus dataset associated with model_run.
    :param model_id: unique identifier of Model Run to which the predictions will be uploaded.
    :param payload: json object representing payload to append operation.  Format should be consistent with
    dataset.append() input documented in nucleus folder.
    :param batch_size: (optional) number of payload items to be uploaded per batch, defaults to 1000.
    :param use_notebook: toggles between normal tqdm bar and tqdm bar for ipynb notebook.
    User should specify use_notebook=True when invoking this function from a notebook environment.
    """
    client = nucleus.NucleusClient(api_key)
    items = payload[ANNOTATIONS_KEY]
    print("uploading {} items...".format(len(items)))
    model_run = client.get_model_run(model_run_id)
    batches = [
        items[i : i + batch_size] for i in range(0, len(items), batch_size)
    ]
    agg_response = {
        DATASET_ID_KEY: dataset_id,
        MODEL_RUN_ID_KEY: model_run_id,
        PREDICTIONS_PROCESSED_KEY: 0,
    }
    tqdm_batches = (
        tqdm.notebook.tqdm(batches) if use_notebook else tqdm.tqdm(batches)
    )
    for batch in tqdm_batches:
        response = model_run.predict({ANNOTATIONS_KEY: batch})
        if STATUS_CODE_KEY in response:
            agg_response[ERRORS_KEY] = response
        else:
            agg_response[PREDICTIONS_PROCESSED_KEY] += response[
                PREDICTIONS_PROCESSED_KEY
            ]
    print("upload complete!")
    return agg_response
