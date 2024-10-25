from logger_config import logger
import itertools
from abc import ABC
import networkx as nx
import pandas as pd
import queue

from rdm.utils import OrderedDictList
from typing import Dict, Optional, List


class PropConfig:
    def __init__(self, tables: Dict, foreign_keys: Dict, primary_keys: Optional[Dict] = None,
                 target_table: Optional[str] = None, target_attribute: Optional[str] = None, max_depth: int = 2):
        self.tables = tables
        self.target_table = target_table
        self.target_attribute = target_attribute
        self.foreign_keys = foreign_keys
        self.primary_keys = primary_keys

        self.core_table = self.tables[target_table] if target_table else None
        self.target_classes = self.core_table[target_attribute] if target_table else None

        self.core_foreign_keys = set()
        self.all_foreign_keys = set()

        self.max_depth = max_depth


class Denormalization(ABC):
    def __init__(self, config: PropConfig, keep_target_table_pk: bool = False):
        self.config = config
        self.fk_graph = self.create_fk_graph(self.config.foreign_keys)
        self.feature_vectors = OrderedDictList()
        self.total_witems = set()
        self.keep_target_table_pk = keep_target_table_pk
        self.parsed_tables = None

    def create_fk_graph(self, foreign_keys):
        graph = nx.Graph()
        for t1, k1, t2, k2 in foreign_keys:
            if t1 == self.config.target_table:
                self.config.core_foreign_keys.add(k1)

            elif t2 == self.config.target_table:
                self.config.core_foreign_keys.add(k2)

            self.config.all_foreign_keys.add(k1)
            self.config.all_foreign_keys.add(k2)
            graph.add_edge(t1, t2, source_column=k1, target_column=k2)
        return graph

    def print_graph(self):
        logger.info("Graph Representation of Foreign Key Relationships:")
        logger.info("\nNodes (Tables):")
        for node in self.fk_graph.nodes():
            print(node)

        logger.info("\nEdges (Foreign Key Relationships):")
        for t1, t2, attributes in self.fk_graph.edges(data=True):
            logger.info(
                f"From {t1} to {t2} - Source Column: {attributes['source_column']}, "
                f"Target Column: {attributes['target_column']}")

    def initialize_queue(self, traversal_map):
        """Initializes the queue with successor tables of the target table."""
        to_traverse = queue.Queue()
        successor_tables = traversal_map.get(self.config.target_table, [])
        # logging.info(f"Successor Tables: {successor_tables}")
        for source_table in successor_tables:
            to_traverse.put((self.config.target_table, 1, source_table))  # queue stores tuples of (table name, depth)
        return to_traverse

    def fill_queue(self, current_table, current_depth, traversal_map, to_traverse):
        """Utility function to fill the queue based on the current table and depth."""
        if current_depth < self.config.max_depth:
            future_tables = traversal_map.get(current_table, [])
            for next_table in future_tables:
                to_traverse.put((current_table, current_depth + 1, next_table))

            # logging.info(f"Queue State: {list(to_traverse.queue)}")
            # logging.info(f"Future tables from {current_table}: {future_tables}")
        return to_traverse

    def traverse_and_fetch_related_data(self) -> pd.DataFrame:
        logger.info("Traversing other tables...")
        traversal_map = dict(nx.bfs_successors(self.fk_graph, self.config.target_table))
        features_data = self.config.core_table.copy()

        self.parsed_tables = {self.config.target_table}  # to avoid future circular join to the target table
        to_traverse = self.initialize_queue(traversal_map=traversal_map)

        while not to_traverse.empty():
            parent_table, current_depth, current_table = to_traverse.get()

            if current_table not in self.parsed_tables:
                self.parsed_tables.add(current_table)
                logger.info(f"Currently applying denormalization over table: {current_table} at depth {current_depth}")
                edge_data = self.fk_graph.get_edge_data(parent_table, current_table)
                source_column, target_column = edge_data['source_column'], edge_data['target_column']
                if source_column not in features_data or source_column in self.config.tables[current_table]:
                    source_column, target_column = target_column, source_column
                features_data = features_data.merge(self.config.tables[current_table],
                                                    how='left',
                                                    left_on=source_column,
                                                    right_on=target_column,
                                                    suffixes=(None, f'__{current_table}'))

                # Extract keys from foreign and primary keys, excluding core foreign keys
                all_keys = set(itertools.chain(self.config.all_foreign_keys, self.config.primary_keys.values()))
                excluded_keys = all_keys - self.config.core_foreign_keys

                # Append '__y' suffix and filter by presence in features_data.columns
                columns_to_drop = {f"{key}__{current_table}" for key in excluded_keys if
                                   f"{key}__{current_table}" in features_data.columns}

                features_data.drop(list(columns_to_drop), axis=1, inplace=True)

                to_traverse = self.fill_queue(current_table=current_table,
                                              current_depth=current_depth,
                                              traversal_map=traversal_map,
                                              to_traverse=to_traverse)

        return features_data

    def clear_columns(self, features: pd.DataFrame):
        """
        Method that will drop all columns related to primary key or foreign key
        """
        if self.keep_target_table_pk:
            available_keys = {key for key in self.config.all_foreign_keys.union(self.config.primary_keys.values())
                              if
                              key in features.columns and key not in self.config.primary_keys[self.config.target_table]}
        else:
            available_keys = {
                key for key in self.config.all_foreign_keys.union(self.config.primary_keys.values())
                if key in features.columns}

        # cols_to_drop = [col for col in features.columns if col in available_keys and col.endswith('__y')]
        cols_to_drop = [col for col in available_keys if col in features.columns]
        features.drop(cols_to_drop, axis=1, inplace=True)

        return features

    def prepare_labels(self, features: pd.DataFrame) -> None:
        self.config.target_classes = features[self.config.target_attribute]
        features.drop(self.config.target_attribute, axis=1, inplace=True)

    def run(self) -> [pd.DataFrame, pd.Series]:
        features_data = self.traverse_and_fetch_related_data()
        features_data = self.clear_columns(features_data)

        self.prepare_labels(features_data)
        return features_data, self.config.target_classes


class Wordification(Denormalization):
    def __init__(self, config: PropConfig, keep_target_table_pk: bool = True):
        super().__init__(config, keep_target_table_pk)
        self.docs = []
        self.table_metadata = None

    def get_data_lineage(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        table_metadata = {}
        excluded_keys = self.config.all_foreign_keys.union(self.config.primary_keys.values())

        for table_name, table in self.config.tables.items():
            if table_name in self.parsed_tables:
                table_metadata[table_name] = [
                    col for col in table.columns
                    if (col in features.columns or f"{col}__{table_name}" in features.columns) and
                       col not in excluded_keys and col not in self.config.target_attribute
                ]

        return table_metadata

    def create_document(self, group: pd.DataFrame) -> [str, str]:
        document_parts = []
        doc_label = group[self.config.target_attribute].values[0]
        group.drop(self.config.target_attribute, axis=1, inplace=True)
        document_parts.extend(self._process_target_table(group))
        document_parts.extend(self._process_related_tables(group))
        return " ".join(document_parts), doc_label

    def _process_target_table(self, group: pd.DataFrame) -> List[str]:
        target_table_name = self.config.target_table
        customer_info = group.iloc[0][self.table_metadata[target_table_name]].to_dict()
        return [f"{target_table_name}_{key}_{value}" for key, value in customer_info.items()]

    def _process_related_tables(self, group: pd.DataFrame) -> List[str]:
        document_parts = []
        for _, row in group.iterrows():
            for table_name, columns in self.table_metadata.items():
                if table_name != self.config.target_table:
                    table_info = row[columns].to_dict()
                    document_parts.extend([f"{table_name}_{key}_{value}" for key, value in table_info.items()])
        return document_parts

    def run(self) -> [List[str], pd.Series]:
        features_data = self.traverse_and_fetch_related_data()
        features_data = self.clear_columns(features_data)
        self.table_metadata = self.get_data_lineage(features_data)

        grouped_data = features_data.groupby(self.config.primary_keys[self.config.target_table])

        documented_data = []
        documents = []
        labels = []

        for customer_id, group in grouped_data:
            current_doc, current_label = self.create_document(group)
            documents.append(current_doc)
            labels.append(current_label)
        return documents, labels
