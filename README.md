# AutoRDM: Automating Relational Data Mining with Efficient Feature Engineering and Machine Learning

---

## Usage

This script allows users to perform machine learning experiments with various classifiers on specified datasets. Before running the script, ensure that the necessary datasets are enabled in the `datasets.yaml` file

### Configuration

#### Datasets Configuration
Edit the `datasets.yaml` file to specify which databases should be used. You must enable the databases by setting `enabled: true` for each dataset you intend to use. Here is an example of how to enable a dataset in the YAML file:

```yaml
datasets:
  binary_classification:
    - sql_type: mysql
      database: null
      target_schema: trains 
      target_table: trains
      target_column: direction
      include_all_schemas: false
      enabled: false
  multiclass_classification:
    - sql_type: mysql
      database: null
      target_schema: genes
      target_table: Classification
      target_column: Localization
      include_all_schemas: false
      enabled: false

  ```

#### Classifiers Configuration
Edit the `default_config.yaml` file or another specified configuration file to customize the settings for the classifiers. Configure the classifier parameters as needed under the `classifiers` section. Here is an example configuration:

```yaml
classifiers:
    random_forest:
      class: sklearn.ensemble.RandomForestClassifier
      param_grid:
        n_estimators: [5, 100]  
        max_depth: [10, 20]  
        min_samples_split: [2, 5]
        min_samples_leaf: [1, 2]
        class_weight: ['balanced']  
  
    extra_trees:
      class: sklearn.ensemble.ExtraTreesClassifier
      param_grid:
        n_estimators: [16, 128]  
        max_depth: [1, 20]  
        min_samples_split: [2, 5]
        min_samples_leaf: [1, 2]
        class_weight: ['balanced']  
  ```

# Benchmark Script: CLI Arguments and Usage

## Overview
This script is used to benchmark different **propositionalization methods** and **classification problem types** on multiple datasets. It takes various parameters as command-line arguments to customize the execution.

## CLI Arguments

| Argument              | Type   | Default Value                                         | Description |
|-----------------------|--------|-----------------------------------------------------|-------------|
| `--results_file`      | string | `"results/debugging_new_datasets_23_06_2024.csv"`   | Path to the results file where the experiment results will be stored. |
| `--classifier_config` | string | `"rdm/classifier_config.yaml"`                      | Path to the YAML configuration file for classifiers. |
| `--fe_config`        | string | `"rdm/fe_config.yaml"`                              | Path to the YAML configuration file for feature engineering transformers. |
| `--folds`            | int    | `10`                                                | Number of folds for cross-validation. |
| `--prop_methods`     | list   | `["denormalization", "wordification"]`             | Propositionalization methods to be used. Available choices: `wordification`, `denormalization`. |
| `--problem_types`    | list   | `["multiclass_classification"]`                     | Types of classification problems to process. Available choices: `binary_classification`, `multiclass_classification`. |

## Example Usage

### 1. Run with default values
This will execute the script with its default settings:
```bash
python benchmark.py
```
### 2. Run with a specific results file and binary classification
```bash
python benchmark.py --results_file results/experiment_results.csv --problem_types binary_classification
```
### 3. Run using only wordification method for multiclass classification with 5 folds
```bash
python benchmark.py --prop_methods wordification --problem_types multiclass_classification --folds 5
```
### 4. Run with a custom classifier and feature engineering configuration
```bash
python benchmark.py --classifier_config custom/classifier.yaml --fe_config custom/fe_config.yaml
```
### 5. Run with multiple problem types and both propositionalization methods
```bash
python benchmark.py --problem_types binary_classification multiclass_classification --prop_methods wordification denormalization
```
### Note

Ensure that all paths and parameters are correctly set according to your environment and requirements.

---
## Use with own data
Current implementation direct connection to your or any public server. Currently, we have connectors to MSSQL Server and MySQL.

The datasets used in this paper can be accessed by connecting to the database available on https://relational-data.org/

Regarding connection to a database, please refer to the `rdm/db_utils.py` script and prepare your own connection details.

## Cite original paper
```
@Article{bdcc8040039,
AUTHOR = {Stanoev, Boris and Mitrov, Goran and Kulakov, Andrea and Mirceva, Georgina and Lameski, Petre and Zdravevski, Eftim},
TITLE = {Automating Feature Extraction from Entity-Relation Models: Experimental Evaluation of Machine Learning Methods for Relational Learning},
JOURNAL = {Big Data and Cognitive Computing},
VOLUME = {8},
YEAR = {2024},
NUMBER = {4},
ARTICLE-NUMBER = {39},
URL = {https://www.mdpi.com/2504-2289/8/4/39},
ISSN = {2504-2289},
ABSTRACT = {With the exponential growth of data, extracting actionable insights becomes resource-intensive. In many organizations, normalized relational databases store a significant portion of this data, where tables are interconnected through some relations. This paper explores relational learning, which involves joining and merging database tables, often normalized in the third normal form. The subsequent processing includes extracting features and utilizing them in machine learning (ML) models. In this paper, we experiment with the propositionalization algorithm (i.e., Wordification) for feature engineering. Next, we compare the algorithms PropDRM and PropStar, which are designed explicitly for multi-relational data mining, to traditional machine learning algorithms. Based on the performed experiments, we concluded that Gradient Boost, compared to PropDRM, achieves similar performance (F1 score, accuracy, and AUC) on multiple datasets. PropStar consistently underperformed on some datasets while being comparable to the other algorithms on others. In summary, the propositionalization algorithm for feature extraction makes it feasible to apply traditional ML algorithms for relational learning directly. In contrast, approaches tailored specifically for relational learning still face challenges in scalability, interpretability, and efficiency. These findings have a practical impact that can help speed up the adoption of machine learning in business contexts where data is stored in relational format without requiring domain-specific feature extraction.},
DOI = {10.3390/bdcc8040039}
}
```