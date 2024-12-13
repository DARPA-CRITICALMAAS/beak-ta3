import numpy as np

from copy import deepcopy
from typing import Tuple, List
from beak.methods.bnn.fastBNN import predict_model
from beak.evaluation.metrics_definitions import classification


def _calculate_metric(y_true, y_pred, metric_function, decimal_places) -> float:
    """
    # TODO: Docstring goes here.
    """
    return round(metric_function(y_true, y_pred), decimal_places)


def binary_classification(
    model: Tuple[List, List, List],
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> List:
    """
    # TODO: Docstring goes here.
    """
    metrics = deepcopy(classification)
    number_classes = np.unique(y).size

    id_loc, fn_loc = "name", "fn"
    metrics_binary_y_pred = ["auc", "auprc"]
    metrics_one_class = ["acc"]

    y_pred, _ = predict_model(model, X)
    y_pred_binarized = np.where(y_pred > threshold, 1, 0)

    if number_classes == 1:
        metrics = [metric for metric in metrics if metric[id_loc] in metrics_one_class]

    for metric in metrics:
        y_eval = y_pred_binarized if not metric[id_loc] in metrics_binary_y_pred else y_pred

        metric.update(
            {
                "value": _calculate_metric(
                    y,
                    y_eval,
                    metric[fn_loc],
                    decimal_places=3
                ),
            }
        )
        metric.pop(fn_loc)

    return metrics
