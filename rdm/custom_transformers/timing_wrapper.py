from logger_config import logger
import time
from sklearn.base import BaseEstimator, TransformerMixin


class TimingWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, name=None):
        self.transformer = transformer
        self.name = name if name else type(transformer).__name__
        self.start_time = None
        self.end_time = None

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def transform(self, X):
        self.start_time = time.time()
        X_transformed = self.transformer.transform(X)
        self.end_time = time.time()

        logger.info(f"{self.name} transformation time: {self.end_time - self.start_time:.4f} seconds")

        return X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        self.start_time = time.time()
        X_transformed = self.transformer.fit_transform(X, y, **fit_params)
        self.end_time = time.time()

        logger.info(f"{self.name} fit_transform time: {self.end_time - self.start_time:.4f} seconds")

        return X_transformed
