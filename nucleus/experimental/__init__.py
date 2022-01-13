"""
# Scale Deploy

Shamelessly copied from PRFAQ :)

Moving an ML model from experiment to production requires significant engineering lift.
An ML engineer needs to package the model into a serving application, provision infrastructure to deploy the application, and then wrestle with automatic scaling, uptime, latency, security, and more once the application is finally deployed.
Leveraging an infrastructure team or serverless compute framework can mitigate infrastructure and deployment issues, but still requires the ML engineer to spend initial effort in the packaging phase.
Once the packaging step is solved, using these standard microservice deployment paradigms can suffice for a prototype, but the result still falls short when moving to the production stage.
This is because standard microservice deployment techniques do not account for the fact that ML models utilize expensive GPUs or have large memory footprints.
Given that ML models are often deployed in business critical scenarios that have stringent latency requirements, failing to account for these factors results in a highly inefficient service that either fails to meet its latency requirements, racks up an excessive infrastructure bill, or both.

Scale Deploy solves these problems by providing ML engineers a simple command line interface for turning a local code snippet into a production service.
First, the ML engineer runs a command to upload their inference code as a bundle, which is an artifact that contains the model weights and the associated preprocessing/postprocessing code.
The command is capable of automatically detecting the user’s code packages. The user then runs a second command to turn any previously uploaded bundle into a service.
The service efficiently utilizes compute resources and automatically scales according to traffic.

# Deploying your model via Scale Deploy

Central to Scale Deploy are the notions of a `ModelBundle` and a `ModelEndpoint`.
A `ModelBundle` consists of a trained model as well as the surrounding preprocessing and postprocessing code.
A `ModelEndpoint` is the compute layer that takes in a `ModelBundle`, and is able to carry out inference requests
by using the `ModelBundle` to carry out predictions. The `ModelEndpoint` also knows infrastructure-level details,
such as how many GPUs are needed, what type they are, how much memory, etc. The `ModelEndpoint` automatically handles
infrastructure level details such as autoscaling and task queueing.

Steps to deploy your model via Scale Deploy:

1. First, you create and upload a `ModelBundle`.

2. Then, you create a `ModelEndpoint`.

3. Lastly, you make requests to the `ModelEndpoint`.

TODO: link some example colab notebook
"""
