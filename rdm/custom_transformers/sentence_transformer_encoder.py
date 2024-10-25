from sklearn.base import TransformerMixin, BaseEstimator
from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the transformer with a specified sentence-transformers model.
        :param model_name: str, the model name from sentence-transformers.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        """
        Fit method, which doesn't need to do anything except return self.
        This is because pre-trained models do not need to fit data.
        :param X: Iterable of texts
        :param y: Ignored
        :return: self
        """
        return self

    def transform(self, X):
        """
        Transform the documents X into embeddings using the pre-trained model.
        :param X: Iterable of texts to encode
        :return: numpy array of embeddings
        """
        return self.model.encode(X, show_progress_bar=True)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Since fit does not do anything, we can directly return transform here.
        :param X: Iterable of texts
        :param y: Ignored
        :return: numpy array of embeddings
        """
        return self.transform(X)
