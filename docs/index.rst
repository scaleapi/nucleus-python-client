Welcome to the Nucleus API Reference!
=====================================

Aggregate metrics in ML are not good enough. To improve production ML, you need to understand qualitative failure modes, fix them by gathering more data, and curate diverse scenarios.

Scale Nucleus helps you:

* Visualize your data
* Curate interesting slices within your dataset
* Review and manage annotations
* Measure and debug your model performance

Nucleus is a new way—the right way—to develop ML models, helping us move away from the concept of one dataset and towards a paradigm of collections of scenarios.

.. _evaluations-v2:

Evaluations V2
--------------

Evaluation V2 runs COCO-style metrics against stored matches (``evaluation_match_v2``) for a **model run**.
Create an evaluation with :meth:`NucleusClient.create_evaluation_v2`; poll with
:meth:`nucleus.evaluation_v2.EvaluationV2.wait_for_completion`; then fetch aggregates via
:meth:`nucleus.evaluation_v2.EvaluationV2.charts` or per-row examples via
:meth:`nucleus.evaluation_v2.EvaluationV2.examples`.

.. code-block:: python

   import nucleus

   client = nucleus.NucleusClient(api_key="YOUR_API_KEY")
   evaluation = client.create_evaluation_v2(
       model_run_id="run_xxx",
       name="my-eval",
       allowed_label_matches=[
           nucleus.AllowedLabelMatch(
               ground_truth_label="car",
               model_prediction_label="vehicle",
           ),
       ],
   )
   evaluation.wait_for_completion()
   charts = evaluation.charts(iou_threshold=0.5)
   fps = evaluation.examples(match_type="FP", limit=20)

The API uses REST endpoints ``/nucleus/modelRun/:id/evaluationsV2``,
``/nucleus/evaluationsV2/:id``, ``/nucleus/evaluationsV2/:id/charts``, and
``POST /nucleus/evaluationsV2/:id/examples``.

.. _installation:

Installation
------------

To use Nucleus, first install it using `pip`:

.. code-block:: console

   (venv) $ pip install scale-nucleus


.. _api:

Sections
--------

.. toctree::
   :maxdepth: 4

   api/nucleus/index
   api/nucleus/metrics/index
   api/nucleus/validate/index


Index
-----

* :ref:`genindex`
