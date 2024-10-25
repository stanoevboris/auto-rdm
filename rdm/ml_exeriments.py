from logger_config import logger
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

from rdm.constants import training_scoring_metrics, testing_scoring_metrics
from rdm.custom_transformers.timing_wrapper import TimingWrapper
from rdm.utils import load_yaml_config


class MLExperiment:
    def __init__(self, feature_config_path: str, classifier_config_path: str, prop_method: str, problem_type: str,
                 dataset: str):
        self.training_scoring_metrics = None
        self.prop_method = prop_method
        self.problem_type = problem_type
        self.dataset = dataset
        self.feature_config = load_yaml_config(feature_config_path)
        self.classifier_config = load_yaml_config(classifier_config_path)

        self.training_scoring_metrics = training_scoring_metrics.get(self.problem_type)
        self.testing_scoring_metrics = testing_scoring_metrics.get(self.problem_type)
        self.refit_metric = self.get_refit_metric()
        self.train_results = {}
        self.test_results = {}

    def get_refit_metric(self):
        if self.problem_type == 'binary_classification':
            return 'roc_auc'
        elif self.problem_type == 'multiclass_classification':
            return 'f1_macro'  # or 'accuracy' or 'roc_auc_ovr'
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    @staticmethod
    def create_pipeline(feature_steps, classifier_info, scoring, refit_metric):
        steps = []
        for step in feature_steps:
            module_path, class_name = step['name'].rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            transformer_class = getattr(module, class_name)

            transformer_params = step.get('params', {})
            transformer = transformer_class(**transformer_params)
            if class_name != 'ConditionalResampler':
                transformer = TimingWrapper(transformer, name=class_name)
            steps.append((class_name.lower(), transformer))

        class_path, class_name = classifier_info['class'].rsplit('.', 1)
        class_module = __import__(class_path, fromlist=[class_name])
        classifier = getattr(class_module, class_name)()

        steps.append(('classifier', classifier))

        pipeline = ImbPipeline(steps)

        param_grid = classifier_info['param_grid']
        search_spaces = {}
        for param, values in param_grid.items():
            param_name = f'classifier__{param}'
            if isinstance(values[0], int):
                search_spaces[param_name] = Integer(min(values), max(values))
            elif isinstance(values[0], float):
                search_spaces[param_name] = Real(min(values), max(values), prior='log-uniform')
            elif isinstance(values[0], str):
                search_spaces[param_name] = Categorical(values)

        stratified_cv = StratifiedKFold(n_splits=10)
        bayes_search = BayesSearchCV(pipeline, search_spaces=search_spaces, n_iter=1, cv=stratified_cv,
                                     scoring=scoring, refit=refit_metric, verbose=10, n_jobs=1, random_state=42)

        return bayes_search

    def run_experiments(self, X, y) -> pd.DataFrame:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        for pipeline_name, feature_steps in self.feature_config[self.prop_method].items():
            for classifier_name, classifier_info in self.classifier_config['classifiers'].items():
                logger.info(f"Creating and evaluating pipeline for {pipeline_name} with {classifier_name}")
                pipeline = self.create_pipeline(feature_steps, classifier_info, self.training_scoring_metrics,
                                                self.refit_metric)
                pipeline.fit(X_train, y_train)
                self.train_results[classifier_name, pipeline_name] = pipeline

                yield self.evaluate_on_test_set(pipeline, X_test, y_test, classifier_name, pipeline_name)

    def evaluate_on_test_set(self, pipeline, X_test, y_test, classifier_name, pipeline_name) -> pd.DataFrame:
        logger.info(f"STARTED: Evaluating test set for {self.dataset} with {pipeline_name}")
        val_predictions = pipeline.predict(X_test)
        val_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None

        df = {}
        for param_name, value in pipeline.best_params_.items():
            df[param_name] = value
        df['classifier'] = classifier_name
        df['feature_engineering_type'] = pipeline_name
        df['dataset'] = self.dataset
        df['problem_type'] = self.problem_type

        for metric_name, (metric_func, kwargs) in self.testing_scoring_metrics.items():
            if metric_name in ['roc_auc', 'average_precision'] and val_proba is not None:
                score = round(metric_func(y_test, val_proba[:, 1], **kwargs), 3)
            else:
                score = round(metric_func(y_test, val_predictions, **kwargs), 3)
            df[metric_name] = score
            print(f'Validation {metric_name} for {classifier_name} with {pipeline_name}: {score}')
        logger.info(f"COMPLETED: Evaluating test set for {self.dataset} with {pipeline_name}")
        return pd.DataFrame.from_dict(df, orient='index').T

    def summarize_train_results(self) -> pd.DataFrame:
        """
        Summarize the results of a GridSearchCV object when refit is set to False.
        """
        logger.info(f"STARTED: Summarizing train set for {self.dataset}")
        classifiers_summaries = []
        for key, bayes_search in self.train_results.items():
            classifier_name, pipeline_name = key
            results_df = pd.DataFrame(bayes_search.cv_results_)

            # Extract parameter and scoring keys
            param_keys = [col for col in results_df.columns if col.startswith('param_')]
            scoring_keys = [col for col in results_df.columns if
                            col.startswith('mean_test_') or col.startswith('std_test_') or col.startswith('rank_test_')]

            important_columns = param_keys + scoring_keys
            current_summary = results_df[important_columns].copy()
            current_summary['classifier'] = classifier_name
            current_summary['feature_engineering_type'] = pipeline_name
            current_summary['dataset'] = self.dataset
            current_summary['problem_type'] = self.problem_type

            # Sort by the first scoring metric rank (you can adjust this as needed)
            first_rank_col = f'rank_test_{self.refit_metric}'
            current_summary_sorted = current_summary.sort_values(by=first_rank_col)
            classifiers_summaries.append(current_summary_sorted)

        logger.info(f"COMPLETED: Summarizing train set for {self.dataset}")
        return pd.concat(classifiers_summaries, ignore_index=True)

    def extract_feature_importance(self):
        feature_importance = {}
        for key, bayes_search in self.train_results.items():
            classifier_name, pipeline_name = key
            model = bayes_search.best_estimator_.named_steps['classifier']

            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = model.coef_[0]
            else:
                continue

            feature_names = [f'Feature {i}' for i in range(len(importance))]
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            feature_importance[(classifier_name, pipeline_name)] = importance_df

        return feature_importance

    @staticmethod
    def plot_feature_importance(feature_importance):
        for key, importance_df in feature_importance.items():
            classifier_name, pipeline_name = key
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.title(f'Feature Importance for {classifier_name} with {pipeline_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'feature_importance_{classifier_name}_{pipeline_name}.png')
            plt.close()

# Usage:
# exp = MLExperiment('feature_engineering.yaml', 'classifiers.yaml')
# experiment_results = exp.run_experiments()
