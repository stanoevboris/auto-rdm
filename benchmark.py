import warnings
import argparse
import time
from rdm.dataset_processor import DatasetProcessor
from rdm.utils import load_yaml_config
from logger_config import logger

warnings.filterwarnings('ignore')


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.config_file = 'rdm/datasets.yaml'

    def load_datasets(self, problem_type: str):
        # Load datasets configuration from YAML or similar
        config = load_yaml_config(self.config_file)

        for dataset in config['datasets'][problem_type]:
            if dataset.get('enabled', True):
                yield dataset

    def check_prop_methods(self):
        if self.args.prop_methods is None:
            raise ValueError('No propositional methods specified')

    def check_problem_types(self):
        if self.args.problem_types is None:
            raise ValueError('No problem types specified')

    def run(self):
        self.check_prop_methods()
        for problem_type in self.args.problem_types:
            for dataset_info in self.load_datasets(problem_type=problem_type):
                processor = DatasetProcessor(dataset_info, self.args, problem_type)
                processor.process()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="results/debugging_new_datasets_23_06_2024.csv",
                        help="Path to the results file")
    parser.add_argument("--classifier_config", default="rdm/classifier_config.yaml",
                        help="Path to the classifiers dataset_config file")
    parser.add_argument("--fe_config", default="rdm/fe_config.yaml",
                        help="Path to the feature engineering transformers config file")
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument(
        "--prop_methods", nargs='*', default=['denormalization', 'wordification'],
        choices=["wordification", "denormalization"])
    parser.add_argument(
        "--problem_types", nargs='*', default=['multiclass_classification'],
        choices=["binary_classification", "multiclass_classification"])
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_arguments()
    benchmark = Benchmark(args)
    benchmark.run()

    end_time = time.time()
    logger.info(f"Experiments total execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
