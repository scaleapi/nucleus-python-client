import tqdm


# TODO: add tqdm for progress updates.
def batch_upload_append(dataset, payload, batch_size=100):
    """
    Takes large payload, splits it into batches, and calls append sequentially.
    """
    batches = get_batches(payload["items"], batch_size)
    for batch in tqdm(batches):
        dataset.append({"items": batch})


def get_batches(items_list, batch_size):
    """
    TODO: add docstring
    """
    return items_list
