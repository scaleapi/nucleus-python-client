Currently, Scale Deploy is still being built out, so the contents of this library are subject to change.

# Scale Deploy

Moving an ML model from experiment to production requires significant engineering lift. 
Scale Deploy provides ML engineers a simple Python interface for turning a local code snippet into a production service.
A ML engineer needs to call a few functions from Scale's SDK, which quickly spins up a production-ready service.
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
