from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    average_precision_score,
)


classification = [
    {
        "short": "acc",
        "name": "Accuracy",
        "description": "The proportion of correct predictions for the positive class out of all predictions.",
        "fn": accuracy_score,
    },
    {
        "short": "bal_acc",
        "name": "Balanced Accuracy",
        "description": "Balanced Accuracy, the average of recall obtained on each class.",
        "fn": balanced_accuracy_score,
    },
    {
        "short": "prec",
        "name": "Precision",
        "description": "The ability of the classifier not to label as positive a sample that is negative.",
        "fn": precision_score,
    },
    {
        "short": "rec",
        "name": "Recall",
        "description": "The ability of the classifier to find all the positive samples.",
        "fn": recall_score,
    },
    {
        "short": "f1",
        "name": "F1 Score",
        "description": "The harmonic mean of precision and recall.",
        "fn": f1_score,
    },
    {
        "short": "mcc",
        "name": "Matthews Correlation Coefficient",
        "description": "Matthews Correlation Coefficient, measures the quality of binary classifications.",
        "fn": matthews_corrcoef,
    },
    {
        "short": "auc",
        "name": "ROC AUC Score",
        "description": "Area Under the Receiver Operating Characteristic Curve (AUROC), measures the ability of a model to distinguish between classes (Note: not intended for imbalanced datasets).",
        "fn": roc_auc_score,
    },
    {
        "short": "auprc",
        "name": "Average Precision Score",
        "description": "Area Under the Precision-Recall Curve, evaluates the trade-off between precision and recall.",
        "fn": average_precision_score,
    },
]
