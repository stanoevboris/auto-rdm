import os
import yaml
from collections import OrderedDict, Counter
from logger_config import logger
import numpy as np

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import KBinsDiscretizer

PROJECT_DIR = os.path.dirname(__file__)

def setup_directory(directory_path):
    """Ensure the directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Directory {directory_path} created.")


def load_yaml_config(config_file: str):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"{config_file} not found.")
        return


def log_dataset_info(train_features, test_features):
    """
    Logs information about the dataset, working with both DataFrames and sparse matrices.
    """
    num_train_records, num_train_features = train_features.shape
    num_test_records, num_test_features = test_features.shape

    total_records = num_train_records + num_test_records

    logger.info(f"Dataset number of records: {total_records}")
    logger.info(f"Train set number of features: {num_train_features}")
    logger.info(f"Test set number of features: {num_test_features}")


def calculate_positive_class_percentage(train_classes, test_classes, representation_type):
    """
    Calculate the percentage occurrence of the positive class in the combined class list or array,
    handling both DataFrames and numpy arrays. The positive class is defined by the 'positive_label'.

    Parameters:
    - train_classes: Training class labels.
    - test_classes: Test class labels.
    - representation_type: Type of representation, affecting how data is processed.

    Returns:
    - The percentage of occurrences of the positive class.
    """
    if representation_type == 'woe':
        # Handling DataFrame: Extracting values as a list
        all_values = list(train_classes.values()) + list(test_classes.values())
    else:
        # Handling numpy array (for sparse matrix, ensure conversion to numpy array beforehand)
        all_values = np.concatenate((train_classes, test_classes))

    # Count occurrences of each class
    occurrences = Counter(all_values)

    # Calculate total number of occurrences
    total_count = sum(occurrences.values())

    # Calculate the percentage of the positive class
    positive_class_count = occurrences.get(1, 0)
    positive_class_percentage = (positive_class_count / total_count) * 100 if total_count > 0 else 0

    return positive_class_percentage


def clear(stx):
    """
    Clean the unnecessary parenthesis
    """

    return stx.replace("`", "").replace("`", "")


def discretize_candidates(df, types, ratio_threshold=0.20, n_bins=20):
    """
    Continuous variables are discrete if more than 30% of the rows are unique.
    """

    ratio_storage = {}
    for enx, type_var in enumerate(types):
        if "int" in type_var or "decimal" in type_var or "float" in type_var:
            ratio_storage = 1. * df[enx].nunique() / df[enx].count()
            if ratio_storage > ratio_threshold and ratio_storage != 1.0:
                to_validate = df[enx].values
                parsed_array = np.array(
                    [np.nan if x == "NULL" else float(x) for x in to_validate])
                parsed_array = interpolate_nans(parsed_array.reshape(-1, 1))
                to_be_discrete = parsed_array.reshape(-1, 1)
                var = KBinsDiscretizer(
                    encode="ordinal",
                    n_bins=n_bins).fit_transform(to_be_discrete)
                df[enx] = var
                if np.isnan(var).any():
                    continue
                df[enx] = df[enx].astype(str)
    return df


class OrderedDictList(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


class OrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = {}
        return self[k]


def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)


def cleanp(stx):
    """
    Simple string cleaner
    """

    return stx.replace("(", "").replace(")", "").replace(",", "")


def interpolate_nans(X):
    """
    Simply replace nans with column means for numeric variables.
    input: matrix X with present nans
    output: a filled matrix X
    """

    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:, j])
        X[mask_j, j] = np.mean(np.flatnonzero(X))
    return X


def is_imbalanced(labels, threshold=0.2) -> bool:
    """
    Check if the dataset is imbalanced based on the provided threshold.

    Parameters:
    - labels (pd.Series): A pandas Series containing the class labels of the dataset.
    - threshold (float): The threshold for determining imbalance.
                            Represents the minimum proportion of the minority class. Defaults to 0.2 (20%).

    Returns:
    - bool: True if the dataset is imbalanced, False otherwise.
    """
    # Calculate the proportion of each class
    class_proportions = labels.value_counts(normalize=True)
    # Find the proportion of the minority class
    min_class_proportion = class_proportions.min()

    return min_class_proportion < threshold


def balance_dataset_with_smote(X_train, y_train):
    """
    Applies SMOTE to balance the training dataset.

    Parameters:
    - X_train (array-like): Training features.
    - y_train (array-like): Training labels.

    Returns:
    - X_resampled (array-like): The resampled training features after applying SMOTE.
    - y_resampled (array-like): The resampled training labels after applying SMOTE.
    """
    # Initialize the SMOTE object
    # smote = SMOTE(random_state=42, sampling_strategy='minority')
    smote_enn = SMOTEENN(smote=SMOTE(random_state=42),
                         enn=EditedNearestNeighbours(sampling_strategy='majority'))
    # Apply SMOTE
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    logger.info(f"Original dataset shape {np.bincount(y_train)}")
    logger.info(f"Resampled dataset shape {np.bincount(y_resampled)}")

    return X_resampled, y_resampled
