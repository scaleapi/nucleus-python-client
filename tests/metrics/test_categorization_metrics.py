from sklearn.metrics import f1_score

from nucleus import CategoryAnnotation, CategoryPrediction
from nucleus.metrics.categorization_metrics import CategorizationF1Metric
from tests.helpers import TEST_CATEGORY_ANNOTATIONS


def test_perfect_match_f1_score():
    annotations = [
        CategoryAnnotation.from_json(p) for p in TEST_CATEGORY_ANNOTATIONS
    ]
    predictions = [
        CategoryPrediction(label=ann.label, reference_id=ann.reference_id)
        for ann in annotations
    ]
    metric = CategorizationF1Metric()
    results = []
    for ann, pred in zip(annotations, predictions):
        results.append(metric.eval([ann], [pred]))

    assert results
    aggregate_result = metric.aggregate(results)
    assert aggregate_result.value == 1


def test_no_match_f1_score():
    annotations = [
        CategoryAnnotation.from_json(p) for p in TEST_CATEGORY_ANNOTATIONS
    ]
    predictions = [
        CategoryPrediction(label="sth_else", reference_id=ann.reference_id)
        for ann in annotations
    ]
    metric = CategorizationF1Metric()
    results = []
    for ann, pred in zip(annotations, predictions):
        results.append(metric.eval([ann], [pred]))

    assert results
    aggregate_result = metric.aggregate(results)
    assert aggregate_result.value == 0


def test_simple_macro_f1():
    """We don't need elaborate f1 testing since we use scikit learn -> Just have one for sanity checking

    Confusion:
    pred: a | b | c  | recall
    gt:  ---|---|--- |
    a     0 | 1 | 0  | 0
    b     0 | 2 | 0  | 1
    c     0 | 0 | 2  | 1
    __________________
    prec: 0 | 0.67 | 1

    sklearn.metrics.classification_report =
                  precision    recall  f1-score   support

               a       0.00      0.00      0.00         1
               b       0.67      1.00      0.80         2
               c       1.00      1.00      1.00         2

        accuracy                           0.80         5
       macro avg       0.56      0.67      0.60         5
    weighted avg       0.67      0.80      0.72         5
    """
    gt = ["a", "b", "b", "c", "c"]
    pr = ["b", "b", "b", "c", "c"]
    macro_f1 = f1_score(gt, pr, average="macro")
    annotations = [
        CategoryAnnotation(label=gt[i], reference_id=f"id_{i}")
        for i in range(len(gt))
    ]
    predictions = [
        CategoryPrediction(label=pr[i], reference_id=f"id_{i}")
        for i in range(len(pr))
    ]
    metric = CategorizationF1Metric()
    results = []
    for ann, pred in zip(annotations, predictions):
        results.append(metric.eval([ann], [pred]))

    aggregate_result = metric.aggregate(results)
    assert aggregate_result.value == macro_f1
