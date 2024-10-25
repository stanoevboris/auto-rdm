import os
import time
from typing import Dict
from logger_config import logger
import numpy as np
import pandas as pd

from rdm.db_utils import get_database
from rdm.ml_exeriments import MLExperiment
from rdm.propositionalization import PropConfig, Wordification, Denormalization

from sklearn.preprocessing import LabelEncoder


class DatasetConfig:
    def __init__(self, dataset_info):
        self.sql_type = dataset_info['sql_type']
        self.database = dataset_info['database']
        self.target_schema = dataset_info['target_schema']
        self.target_table = dataset_info['target_table']
        self.target_attribute = dataset_info['target_column']
        self.include_all_schemas = dataset_info['include_all_schemas']
        self.max_depth = 2


class DatasetProcessor:
    def __init__(self, dataset_info: Dict, args, problem_type: str):
        self.dataset_config = DatasetConfig(dataset_info)
        self.args = args
        self.problem_type = problem_type
        self.tables = None
        self.primary_keys = None
        self.foreign_keys = None

    @staticmethod
    def clean_dataframes(tables: dict):
        for table_name, df in tables.items():
            # remove all columns with null values percentage above 80%
            tables[table_name] = df.loc[:, df.isnull().mean() < .8]

        return tables

    def preprocess_tables(self) -> None:
        self.tables = self.clean_dataframes(tables=self.tables)
        if self.dataset_config.target_schema == 'AdventureWorks2014':
            soh = self.tables['SalesOrderHeader'].copy()
            soh['previous_order_date'] = soh.groupby('CustomerID')['OrderDate'].shift(1)
            soh['days_without_order'] = (soh['OrderDate'] - soh['previous_order_date']).dt.days.fillna(0)
            cut_off_date = soh['OrderDate'].max() - pd.DateOffset(days=180)

            def calculate_churn(row):
                if row['OrderDate'] >= cut_off_date:
                    return None
                elif row['days_without_order'] <= 180:
                    return 0
                else:
                    return 1

            soh['churn'] = soh.apply(calculate_churn, axis=1)

            # Reset the index
            soh = soh[soh['churn'].notna()]
            soh.reset_index(drop=True, inplace=True)
            soh['churn'] = soh['churn'].astype(np.int64)
            # soh['SalesPersonID'] = soh['churn'].astype(np.int64)
            soh.drop(['previous_order_date', 'days_without_order'], axis=1, inplace=True)

            self.tables['SalesOrderHeader'] = soh.copy()
            self.foreign_keys.remove(['SalesOrderHeader', 'SalesPersonID', 'SalesPerson', 'BusinessEntityID'])
            self.foreign_keys.remove(['Customer', 'StoreID', 'Store', 'BusinessEntityID'])
        elif self.dataset_config.target_schema == 'imdb_ijs':
            from rdm.imdb_movies_constants import top_250_movies, bottom_100_movies
            movies = self.tables['movies'].copy()
            positive_df = pd.DataFrame(top_250_movies, columns=["name", "year", "label"])
            negative_df = pd.DataFrame(bottom_100_movies, columns=["name", "year", "label"])
            result_df = pd.concat([positive_df, negative_df], ignore_index=True)
            result_df["year"] = result_df["year"].astype(int)
            result_with_original_data = movies.merge(result_df, on=["name", "year"], how="inner")
            movies = result_with_original_data[["id", "name", "year", "label"]]
            self.tables["movies"] = movies.copy()
        elif self.dataset_config.target_schema == 'AdventureWorks2014':
            self.foreign_keys.remove(['SalesOrderHeader', 'SalesPersonID', 'SalesPerson', 'BusinessEntityID'])
        elif self.dataset_config.target_schema == 'financial' and self.problem_type == 'binary_classification':
            loan = self.tables['loan'].copy()
            loan = loan[loan['status'].isin(['A', 'B'])]
            self.tables['loan'] = loan.copy()
        elif self.dataset_config.target_schema == 'ftp':
            session = self.tables['session'].copy()
            session = session[session['gender'].isin(['male', 'female'])]
            self.tables['session'] = session.copy()
        elif self.dataset_config.target_schema == 'Hockey':
            master = self.tables['Master'].copy()
            master = master[master['shootCatch'].isin(['L', 'R', 'B'])]
            self.tables['Master'] = master.copy()
            self.foreign_keys.remove(['Master', 'coachID', 'Coaches', 'coachID'])
            self.foreign_keys.remove(['AwardsCoaches', 'coachID', 'Coaches', 'coachID'])
        elif self.dataset_config.target_schema == 'genes':
            classification = self.tables['Classification'].copy()
            classification = classification[~classification['Localization'].isin(['integral membrane', 'endosome',
                                                                                  'extracellular', 'cell wall',
                                                                                  'lipid particles', 'peroxisome',
                                                                                  'vacuole', 'transport vesicles'])]
            self.tables['Classification'] = classification.copy()
            # drop Localization from Genes tables since it causes data leak
            genes = self.tables.get('Genes').copy()
            genes.drop(['Localization'], axis=1, inplace=True)
            self.tables['Genes'] = genes.copy()

        elif self.dataset_config.target_schema == 'medical':
            examination = self.tables['Examination'].copy()
            examination = examination[~examination['Thrombosis'].isin([3])]
            self.tables['Examination'] = examination.copy()
            self.primary_keys['Examination'] = "ID"

    def process(self):
        logger.info(f"Processing dataset: {self.dataset_config.target_schema},"
                     f" Table: {self.dataset_config.target_table}")
        start_time = time.time()
        try:
            db_object = get_database(sql_type=self.dataset_config.sql_type,
                                     database=self.dataset_config.database,
                                     target_schema=self.dataset_config.target_schema,
                                     include_all_schemas=self.dataset_config.include_all_schemas)
            self.tables, self.primary_keys, self.foreign_keys = db_object.get_data()
            self.preprocess_tables()
            self.evaluate(self.tables, self.primary_keys, self.foreign_keys)
        finally:
            end_time = time.time()
            logger.info(
                f"Dataset: {self.dataset_config.target_schema} - Execution time: {end_time - start_time:.4f} seconds")

    def propositionalize(self, method, tables, primary_keys, foreign_keys, target_table, target_attribute):
        methods = {
            "wordification": Wordification,
            "denormalization": Denormalization
        }
        prop_config = PropConfig(tables=tables, primary_keys=primary_keys, foreign_keys=foreign_keys,
                                 target_table=target_table, target_attribute=target_attribute,
                                 max_depth=self.dataset_config.max_depth)

        prop_method = methods.get(method)
        prop_object = prop_method(config=prop_config)
        return prop_object.run()

    @staticmethod
    def encode_labels(labels: pd.Series):
        le = LabelEncoder()
        return le.fit_transform(labels)

    def evaluate(self, tables, primary_keys, foreign_keys):
        for method in self.args.prop_methods:
            features, labels = self.propositionalize(method, tables, primary_keys, foreign_keys,
                                                     target_table=self.dataset_config.target_table,
                                                     target_attribute=self.dataset_config.target_attribute)
            labels = self.encode_labels(labels)
            exp = MLExperiment(feature_config_path=self.args.fe_config,
                               classifier_config_path=self.args.classifier_config, prop_method=method,
                               problem_type=self.problem_type,
                               dataset=self.dataset_config.target_schema)
            testing_results_dfs = [df for df in exp.run_experiments(features, labels)]
            testing_results = pd.concat(testing_results_dfs, ignore_index=True)
            training_results = exp.summarize_train_results()
            # Check if the file exists
            training_results_file = f"{self.args.results_file}_training_results.csv"
            testing_results_file = f"{self.args.results_file}_testing_results.csv"
            train_results_file_exists = os.path.isfile(training_results_file)
            test_results_file_exists = os.path.isfile(testing_results_file)

            # Append to the file if it exists, write headers only if the file does not exist
            training_results.to_csv(training_results_file, mode='a', index=False, header=not train_results_file_exists)
            testing_results.to_csv(testing_results_file, mode='a', index=False, header=not test_results_file_exists)
