import torch
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from torch_geometric.data import Data


def evaluate_node_classification(z: torch.Tensor, data: Data) -> dict:
    metrics = {}
    lp_model = sk_lm.LogisticRegression(n_jobs=-1, max_iter=250)

    for split in ("train", "val", "test"):
        mask = data[f"{split}_mask"]
        y_true = data.y[mask]

        if split == "train":
            lp_model.fit(X=z[mask], y=y_true)

        y_score = lp_model.predict_proba(X=z[mask])
        y_pred = lp_model.predict(X=z[mask])

        metrics[split] = calc_metrics(
            y_score=y_score,
            y_pred=y_pred,
            y_true=y_true,
            max_cls=data.y.max() + 1,
        )

    return metrics


def calc_metrics(y_score, y_pred, y_true, max_cls):
    """Composes multiple metrics from Scikit-learn into single dictionary.

    The computed metrics are: precision, recall, f1 per class, the AUC score
    and the confusion matrix.
    """
    metrics = {
        **sk_mtr.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
        ),
        "auc": sk_mtr.roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            average="macro",
            multi_class="ovr",
            labels=range(max_cls),
        ),
        "cm": sk_mtr.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=range(max_cls),
        ).tolist(),
    }

    return metrics
