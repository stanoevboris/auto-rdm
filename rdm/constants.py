from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    matthews_corrcoef, average_precision_score, cohen_kappa_score, hamming_loss, log_loss

training_scoring_metrics = {
    'binary_classification': {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score),
        'roc_auc': make_scorer(roc_auc_score, response_method=("decision_function", "predict_proba")),
        'mcc': make_scorer(matthews_corrcoef),
        'average_precision': make_scorer(average_precision_score)
    },
    'multiclass_classification': {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_micro': make_scorer(f1_score, average='micro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'precision_micro': make_scorer(precision_score, average='micro'),
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'recall_micro': make_scorer(recall_score, average='micro'),
        'recall_weighted': make_scorer(recall_score, average='weighted'),
        # 'roc_auc_ovr': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True),
        # 'average_precision': make_scorer(average_precision_score, needs_proba=True),
        'cohen_kappa': make_scorer(cohen_kappa_score),
        'hamming_loss': make_scorer(hamming_loss, greater_is_better=False),
    }
}

testing_scoring_metrics = {
    "binary_classification": {
        'accuracy': (accuracy_score, {}),
        'f1': (f1_score, {'average': 'weighted'}),
        'f1_weighted': (f1_score, {}),
        'precision': (precision_score, {'zero_division': 0}),
        'recall': (recall_score, {}),
        'roc_auc': (roc_auc_score, {}),
        'mcc': (matthews_corrcoef, {}),
        'average_precision': (average_precision_score, {})},
    "multiclass_classification": {
        'accuracy': (accuracy_score, {}),
        'f1_macro': (f1_score, {'average': 'macro'}),
        'f1_micro': (f1_score, {'average': 'micro'}),
        'f1_weighted': (f1_score, {'average': 'weighted'}),
        'precision_macro': (precision_score, {'average': 'macro'}),
        'precision_micro': (precision_score, {'average': 'micro'}),
        'precision_weighted': (precision_score, {'average': 'weighted'}),
        'recall_macro': (recall_score, {'average': 'macro'}),
        'recall_micro': (recall_score, {'average': 'micro'}),
        'recall_weighted': (recall_score, {'average': 'weighted'}),
        # 'roc_auc_ovr': (roc_auc_score, {'multi_class': 'ovr', 'needs_proba': True}),
        'cohen_kappa': (cohen_kappa_score, {}),
        'hamming_loss': (hamming_loss, {}),
    }
}
