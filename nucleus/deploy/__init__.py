"""

Moving an ML model from experiment to production requires significant engineering lift.
Scale Deploy provides ML engineers a simple Python interface for turning a local code snippet into a production service.
A ML engineer simply needs to call a few functions from Scale's SDK, which quickly spins up a production-ready service.
The service efficiently utilizes compute resources and automatically scales according to traffic.


Central to Scale Deploy are the notions of a `ModelBundle` and a `ModelEndpoint`.

A `ModelBundle` consists of a trained model as well as the surrounding preprocessing and postprocessing code.
Specifically, a `ModelBundle` consists of two Python objects, a `load_predict_fn`, and either a `model` or `load_model_fn`; such that


    load_predict_fn(model)


or


    load_predict_fn(load_model_fn())


returns a function `predict_fn` that takes in one argument representing model input,
and outputs one argument representing model output.

Typically, a `model` would be a Pytorch nn.Module or Tensorflow Keras model.

.. image:: /../docs/images/deploy/model_bundle.png
    :width: 200px

TODO should we include a specific example here?

A `ModelEndpoint` is the compute layer that takes in a `ModelBundle`, and is able to carry out inference requests
by using the `ModelBundle` to carry out predictions. The `ModelEndpoint` also knows infrastructure-level details,
such as how many GPUs are needed, what type they are, how much memory, etc. The `ModelEndpoint` automatically handles
infrastructure level details such as autoscaling and task queueing. There are two types of `ModelEndpoint`s:
`SyncModelEndpoint`s and `AsyncModelEndpoint`s.

.. image:: /../docs/images/deploy/model_endpoint.png
    :width: 400px

A `SyncModelEndpoint` takes in requests and immediately returns the response in a blocking manner.
The `SyncModelEndpoint` always consumes resources, and autoscales on the number of inflight requests.

An `AsyncModelEndpoint` takes in requests and returns an asynchronous response token. The user can later query to monitor
the status of the request. We may later expose a callback mechanism. Asynchronous endpoints can scale up from zero,
which make them a cost effective choice for services that are not latency sensitive.
Asynchronous endpoints autoscale on the number of inflight requests.

In addition, we will expose another abstraction called a `BatchJob`, which takes in a `ModelBundle` and a list of inputs
to predict on. Once a batch job completes, it cannot be restarted or accept additional requests.
Deploy maintains metadata about batch jobs for users to query, even after batch jobs are complete.

Choosing between different types of inference:

`SyncModelEndpoints` are good if:

- You have strict latency requirements (e.g. on the order of seconds or less)

- You are willing to have resources continually allocated

`AsyncModelEndpoints` are good if:

- You want to save on compute costs

- Your inference code takes a long time to run

- Your latency requirements are on the order of minutes.

`BatchJobs` are good if:

- You know there is a large batch of inputs ahead of time

- You want to process data in an offline fashion

Steps to deploy your model via Scale Deploy:

1. First, you create and upload a `ModelBundle`. Pass your trained model as well as pre-/post-processing code to
the Scale Deploy Python SDK, and we'll create a model bundle based on the code and store it in our Bundle Store.

2. Then, you create a `ModelEndpoint`. Pass a `ModelBundle` as well as infrastructure settings such as #GPUs to our SDK.
This provisions resources on Scale's cluster dedicated to your `ModelEndpoint`.

3. Lastly, you make requests to the `ModelEndpoint`. You can make requests through the Python SDK, or make HTTP requests directly
to Scale.

.. image:: /../docs/images/deploy/request_lifecycle.png
    :width: 400px

TODO: link some example colab notebook
"""

from .client import DeployClient
from .model_bundle import ModelBundle
from .model_endpoint import AsyncModelEndpoint, AsyncModelEndpointBatchResponse
