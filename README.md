# AutoRDM: Automating Relational Data Mining with Efficient Feature Engineering and Machine Learning

---

## Usage

This script allows users to perform machine learning experiments with various classifiers on specified datasets. Before running the script, ensure that the necessary datasets are enabled in the `datasets.yaml` file

### Configuration

#### Datasets Configuration
Edit the `datasets.yaml` file to specify which databases should be used. You must enable the databases by setting `enabled: true` for each dataset you intend to use. Here is an example of how to enable a dataset in the YAML file:

```yaml
datasets:
  - sql_type: mssql
    database: woe
    target_schema: indoor_user_movement
    target_table: target
    target_column: class_label
    include_all_schemas: false
    enabled: true  # Set to true to enable this dataset
```

#### Classifiers Configuration
Edit the `default_config.yaml` file or another specified configuration file to customize the settings for the classifiers. Configure the classifier parameters as needed under the `classifiers` section. Here is an example configuration:

```yaml
classifiers:
  - name: random_forest_learner
    params:
      - n_estimators: [100, 200]
        max_depth: [null]
        min_samples_split: [2]
        min_samples_leaf: [1]
        representation_type: ["sklearn_tfidf"]

  - name: extra_tree_learner
    params:
      - n_estimators: [100, 300]
        max_depth: [null]
        min_samples_split: [2]
        min_samples_leaf: [1]
        representation_type: [sklearn_tfidf"]
```

### Command-line Arguments

The script accepts several command-line arguments to customize its execution:

- `--results_file`: Specifies the path to the results file where the output will be stored. Default is `experiments.csv`.
- `--config_file`: Specifies the path to the configuration file for classifiers. Default is `default_config.yaml`.
- `--folds`: Specifies the number of folds for cross-validation. Default is 10.

### Running the Script

To run the script with default settings, simply execute:

```bash
python benchmark_original.py
```

To specify a different results file and config file, and to change the number of folds used in the experiment, you can run:

```bash
python benchmark_original.py --results_file="my_results.csv" --config_file="my_config.yaml" --folds=5
```

This command will direct the script to use `my_results.csv` as the results file, `my_config.yaml` as the configuration file, and perform 5-fold cross-validation.

### Note

Ensure that all paths and parameters are correctly set according to your environment and requirements.

---
## Use with own data
Current implementation direct connection to your or any public server. Currently, we have connectors to MSSQL Server and MySQL.

The datasets used in this paper can be accessed by connecting to the database available on https://relational-data.org/

Regarding connection to a database, please refer to the `rdm/db_utils.py` script and prepare your own connection details.
