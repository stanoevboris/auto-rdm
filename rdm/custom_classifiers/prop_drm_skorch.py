import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import torch.nn as nn
from skorch import NeuralNetClassifier

torch.manual_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DRMArchitecture(nn.Module):
    def __init__(self, input_size, dropout=0.1, hidden_layer_size=10, output_neurons=1):
        super(DRMArchitecture, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(hidden_layer_size, 16),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(16, output_neurons),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class PropDRM(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=8, num_epochs=10, learning_rate=0.0001, patience=5, hidden_layer_size=30,
                 dropout=0.2):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Capture all unique classes
        X, y = check_X_y(X, y, accept_sparse=True, dtype='numeric', force_all_finite='allow-nan')
        input_size = X.shape[1]
        self.net = NeuralNetClassifier(
            DRMArchitecture(input_size, dropout=self.dropout, hidden_layer_size=self.hidden_layer_size,
                            output_neurons=1),
            max_epochs=self.num_epochs,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            device=self.device,
            criterion=nn.BCELoss,

        )
        self.net.fit(X.astype(np.float32), y.astype(np.float32).reshape(-1, 1))
        return self

    def predict(self, X):
        check_is_fitted(self, 'net')
        X = check_array(X, accept_sparse=True)
        return self.net.predict(X.astype(np.float32))

    def predict_proba(self, X):
        check_is_fitted(self, 'net')
        X = check_array(X, accept_sparse=True)
        return self.net.predict_proba(X.astype(np.float32))

    # def score(self, X, y, **kwargs):
    #     check_is_fitted(self, 'net')
    #     y_pred_proba = self.predict_proba(X)
    #     return roc_auc_score(y, y_pred_proba[:, 1])
