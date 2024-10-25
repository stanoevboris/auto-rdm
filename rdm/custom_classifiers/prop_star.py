import numpy as np
import os
import subprocess
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
import time


class PropStar(BaseEstimator, ClassifierMixin):
    """
    This is a simple wrapper for the StarSpace learner modified to be compatible with scikit-learn.
    """

    def __init__(self, binary="./bin/starspace", tmp_folder="tmp",
                 epoch=5, dim=100, learning_rate=0.01, neg_search_limit=50,
                 max_neg_samples=10, verbose=0, model_path="./tmp/storedModel"):
        self.binary = binary
        self.tmp = tmp_folder
        self.epoch = epoch
        self.dim = dim
        self.lr = learning_rate
        self.nsl = neg_search_limit
        self.nspb = max_neg_samples
        self.verbose = verbose
        self.model_path = model_path
        self.parameter_string = (f"-epoch {self.epoch} -dim {self.dim} -lr {self.lr} "
                                 f"-negSearchLimit {self.nsl} -maxNegSamples {self.nspb} -verbose {self.verbose}")

    @staticmethod
    def data_to_text(data, label_tag=False):
        if label_tag:
            return ["__label__" + str(x) for x in data]
        else:
            # Assuming data is a sparse matrix or similar
            return [" ".join(map(str, row.nonzero()[0])) for row in data]

    def write_to_tmp(self, data, tag="train"):
        with open(os.path.join(self.tmp, f"{tag}_data.txt"), "w") as file:
            file.write("\n".join(data))

    def call_starspace_binary(self, train=True):
        if train:
            command = (f"{self.binary} train {self.parameter_string} "
                       f"-trainFile {self.tmp}/train_data.txt -model {self.model_path}")
        else:
            command = (f"{self.binary} test -model {self.model_path} "
                       f"-testFile {self.tmp}/test_data.txt -predictionFile {self.tmp}/predictions.txt")
        os.system(command)

    def fit(self, X, y):
        train_data = self.data_to_text(X)
        train_labels = self.data_to_text(y, label_tag=True)
        train_examples = [f"{d} {l}" for d, l in zip(train_data, train_labels)]
        self.write_to_tmp(train_examples, tag="train")
        self.call_starspace_binary(train=True)
        return self

    def predict(self, X):
        test_data = self.data_to_text(X)
        self.write_to_tmp(test_data, tag="test")
        self.call_starspace_binary(train=False)
        # Assuming predictions.txt is formatted correctly
        with open(os.path.join(self.tmp, "predictions.txt"), "r") as file:
            predictions = [int(line.strip().split()[-1].replace("__label__", "")) for line in file]
        return np.array(predictions)

    def score(self, X, y, **kwargs):
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def cleanup(self):
        os.system(f"rm -rf {self.tmp}/*")
