# Notes on defining Evaluation Criteria for a test

To define the evaluation criteria for a scenario test we've created some syntactic sugar to make it look closer to an
actual function call, and we also hide away implementation details related to our data model that simply are not clear,
UX-wise.

Instead of defining criteria like this:

```python
from nucleus.modelci.data_transfer_objects.eval_function import (
    EvaluationCriterion,
    ThresholdComparison,
)

criteria = [
    EvaluationCriterion(
        eval_function_id="ef_c6m1khygqk400918ays0",  # bbox_recall
        threshold_comparison=ThresholdComparison.GREATER_THAN,
        threshold=0.5,
    ),
]
```

we define it like this:

```python

bbox_recall = client.modelci.eval_functions.bbox_recall
criteria = [
    bbox_recall() > 0.5
]
```

The chosen method allows us to document the available evaluation functions in an IDE friendly fashion and hides away 
details like internal IDs ("ef_....").

The actual `EvaluationCriterion` is created by overloading the comparison operators for the base class of an evaluation
function. Instead of the comparison returning a bool, we've made it create an `EvaluationCriterion` with the correct 
signature to send over the wire to our API.
