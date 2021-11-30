import os
import time
from itertools import zip_longest

from absl import app, flags

import nucleus
from nucleus import annotation
from nucleus.dataset import Dataset
from nucleus.job import JobError

FLAGS = flags.FLAGS

IMAGE_URL = "https://github.com/scaleapi/nucleus-python-client/raw/master/tests/testdata/airplane.jpeg"

# Global flags
flags.DEFINE_string(
    "api_key",
    os.environ["NUCLEUS_PYTEST_API_KEY"],
    "API Key to use. Defaults to NUCLEUS_PYTEST_API_KEY environment variable",
)

flags.DEFINE_integer("job_parallelism", 8, "Amount of concurrent jobs to use.")

# Dataset upload flags
flags.DEFINE_enum(
    "create_or_reuse_dataset",
    "create",
    ["create", "reuse"],
    "If create, upload a new dataset. If reuse, use the dataset id flag to reuse an existing dataset.",
)
flags.DEFINE_string(
    "dataset_id",
    "ds_c4vbhzfp7h9g0d16hwh0",  # Dataset with 1M items
    "Dataset id to use if reusing an existing dataset",
)
flags.DEFINE_integer(
    "num_dataset_items",
    100,
    "Number of dataset items to create if creating a dataset",
    lower_bound=0,
)
flags.DEFINE_bool(
    "cleanup_dataset", True, "Whether to delete the dataset after the test."
)

# Annotation upload flags
flags.DEFINE_integer(
    "num_annotations_per_dataset_item",
    1,
    "Number of annotations per dataset item",
    lower_bound=1,
)

# Prediction upload flags
flags.DEFINE_integer(
    "num_predictions_per_dataset_item",
    1,
    "Number of annotations per dataset item",
    lower_bound=0,
)

TIMINGS = {}


def chunk(iterable, chunk_size, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * chunk_size

    for chunk_iterable in zip_longest(*args, fillvalue=fillvalue):
        yield filter(lambda x: x is not None, chunk_iterable)


def client():
    return nucleus.NucleusClient(api_key=FLAGS.api_key)


def generate_fake_metadata(index):
    return {
        "categorical": str(index % 100),
        "string": str(index),
        "number": index,
    }


def dataset_item_generator():
    for i in range(FLAGS.num_dataset_items):
        yield nucleus.DatasetItem(
            image_location=IMAGE_URL,
            reference_id=str(i),
            metadata=generate_fake_metadata(i),
            upload_to_scale=False,
        )


def get_num_dataset_items():
    if FLAGS.create_or_reuse_dataset == "create":
        num_dataset_items = FLAGS.num_dataset_items
    else:
        num_dataset_items = (
            client().get_dataset(FLAGS.dataset_id).info()["length"]
        )
    return num_dataset_items


def annotation_generator():
    for i in range(get_num_dataset_items()):
        for j in range(FLAGS.num_annotations_per_dataset_item):
            yield nucleus.BoxAnnotation(
                label=str(j % 10),
                x=j,
                y=j,
                width=10,
                height=10,
                reference_id=str(i),
                metadata=generate_fake_metadata(j),
            )


def prediction_generator():
    for i in range(get_num_dataset_items()):
        for j in range(FLAGS.num_predictions_per_dataset_item):
            yield nucleus.BoxPrediction(
                label=str(j % 10),
                x=j,
                y=j,
                width=10,
                height=10,
                reference_id=str(i),
                metadata=generate_fake_metadata(j),
                confidence=1,
            )


def create_or_get_dataset():
    if FLAGS.create_or_reuse_dataset == "create":
        dataset = client().create_dataset("Privacy Mode Load Test Dataset")
        print("Starting dataset item upload")
        tic = time.time()
        chunk_size = FLAGS.num_dataset_items // FLAGS.job_parallelism
        jobs = []
        for dataset_item_chunk in chunk(dataset_item_generator(), chunk_size):
            jobs.append(
                dataset.append(
                    dataset_item_chunk, update=True, asynchronous=True
                )
            )

        for job in jobs:
            try:
                job.sleep_until_complete(False)
            except JobError:
                print(job.errors())
        toc = time.time()
        print("Finished dataset item upload: %s" % (toc - tic))
        TIMINGS[f"Dataset Item Upload {FLAGS.num_dataset_items}"] = toc - tic
    else:
        print(f"Reusing dataset {FLAGS.dataset_id}")
        dataset = client().get_dataset(FLAGS.dataset_id)
    return dataset


def upload_annotations(dataset: Dataset):
    print("Starting annotation upload")
    tic = time.time()
    jobs = []
    num_annotations = (
        FLAGS.num_dataset_items * FLAGS.num_annotations_per_dataset_item
    )
    chunk_size = num_annotations // FLAGS.job_parallelism
    for annotation_chunk in chunk(annotation_generator(), chunk_size):
        jobs.append(
            dataset.annotate(
                list(annotation_chunk), update=False, asynchronous=True
            )
        )

    for job in jobs:
        try:
            job.sleep_until_complete(False)
        except JobError:
            print(job.errors())
    toc = time.time()
    print("Finished annotation upload: %s" % (toc - tic))
    TIMINGS[f"Annotation Upload {num_annotations}"] = toc - tic


def upload_predictions(dataset: Dataset):
    model = client().add_model(
        name="Load test model", reference_id="model_" + str(time.time())
    )
    run = model.create_run(
        name="Test model run", dataset=dataset, predictions=[]
    )

    tic = time.time()

    print("Starting prediction upload")

    num_predictions = (
        FLAGS.num_dataset_items * FLAGS.num_predictions_per_dataset_item
    )
    chunk_size = num_predictions // FLAGS.job_parallelism
    jobs = []
    for prediction_chunk in chunk(prediction_generator(), chunk_size):
        jobs.append(
            run.predict(list(prediction_chunk), update=True, asynchronous=True)
        )

    for job in jobs:
        try:
            job.sleep_until_complete(False)
        except JobError:
            print(job.errors())
    toc = time.time()
    print("Finished prediction upload: %s" % (toc - tic))
    TIMINGS[f"Prediction Upload {num_predictions}"] = toc - tic


def main(unused_argv):
    dataset = create_or_get_dataset()
    try:
        upload_annotations(dataset)
    except Exception as e:
        print(e)

    try:
        upload_predictions(dataset)
    except Exception as e:
        print(e)

    if FLAGS.cleanup_dataset and FLAGS.create_or_reuse_dataset == "create":
        client().delete_dataset(dataset.id)

    print(TIMINGS)


if __name__ == "__main__":
    app.run(main)
