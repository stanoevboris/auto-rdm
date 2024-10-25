import pandas as pd
from sqlalchemy import text
from rdm.db_utils import create_connection


def create_mssql_schema(connection, database_name: str):
    connection.execute(text(
        f"IF (SCHEMA_ID('{database_name}') IS NOT NULL)"
        f" BEGIN "
        f"DROP SCHEMA [{database_name}];"
        f" END "
    ))
    connection.execute(text(f"CREATE SCHEMA [{database_name}] AUTHORIZATION [dbo]"))


class Normalize:
    def __init__(self, denormalized_table: pd.DataFrame, denormalized_table_name: str):
        self.denormalized_table = denormalized_table
        self.entity_set = {denormalized_table_name: self.denormalized_table.copy()}
        self.entity_set[denormalized_table_name]['id'] = self.denormalized_table.index
        self.primary_keys = {denormalized_table_name: 'id'}
        self.relations = {}

    def create_entity(self, source_entity_name: str,
                      target_entity_name: str,
                      columns: list,
                      distinct_values: bool,
                      retain_columns=None):
        """
        Method that creates table with extracting columns from the source table.
         It also creates relation between the two tables
        :param source_entity_name:
        :param target_entity_name:
        :param columns:
        :param distinct_values:
        :param retain_columns:
        :return:
        """
        if retain_columns is None:
            retain_columns = []
        if source_entity_name not in self.entity_set:
            raise ValueError("Given source entity is not present in the current entity set!")

        source_df = self.entity_set[source_entity_name]
        target_entity_pk = f'{target_entity_name}_id'
        drop_columns = list(set(columns) - set(retain_columns))
        if source_entity_name not in self.relations:
            self.relations[source_entity_name] = {}
        if target_entity_name not in self.relations:
            self.relations[target_entity_name] = {}

        if not distinct_values:
            self.entity_set[target_entity_name] = source_df[[*columns]].copy()
            self.entity_set[target_entity_name][target_entity_pk] = source_df.index
            self.entity_set[target_entity_name].set_index(target_entity_pk)

            source_entity_pk = self.primary_keys[source_entity_name]
            self.primary_keys[target_entity_name] = target_entity_pk
            self.relations[source_entity_name][target_entity_name] = (source_entity_pk, target_entity_pk)

            source_df.drop(labels=[*drop_columns], axis=1, inplace=True)
        else:
            self.entity_set[target_entity_name] = source_df[[*columns]].drop_duplicates().reset_index(drop=True)
            self.entity_set[target_entity_name][target_entity_pk] = \
                range(len(self.entity_set[target_entity_name]))
            self.entity_set[target_entity_name].set_index(target_entity_pk)

            self.entity_set[source_entity_name] = source_df \
                .merge(self.entity_set[target_entity_name], on=[*columns], how='left') \
                .drop(labels=[*drop_columns], axis=1)

            source_entity_fk = target_entity_pk
            self.primary_keys[target_entity_name] = target_entity_pk
            self.relations[target_entity_name][source_entity_name] = (target_entity_pk, source_entity_fk)

    def remove_entity(self, entity_name: str):
        self.entity_set.pop(entity_name, f"{entity_name} key not found!")
        self.primary_keys.pop(entity_name, f"{entity_name} key not found!")
        self.relations.pop(entity_name, f"{entity_name} key not found!")
        # delete occurrence in other relations
        for entity in self.relations.keys():
            self.relations[entity].pop(entity_name, f"{entity_name} key not found!")

    def add_custom_relation(self, parent_table_name: str,
                            parent_table_column: str,
                            child_table_name: str,
                            child_table_column: str):
        if parent_table_name not in self.relations:
            self.relations[parent_table_name] = {}
        self.relations[parent_table_name][child_table_name] = (parent_table_column, child_table_column)

    def persist_entity_set(self, database_name: str):
        engine = create_connection(database_name="woe")
        with engine.connect() as connection:
            create_mssql_schema(connection=connection, database_name=database_name)
            for entity in self.entity_set.keys():
                db_entity = f"{database_name}.{entity}"
                self.entity_set[entity].to_sql(entity,
                                               connection,
                                               schema=database_name,
                                               if_exists="replace",
                                               index=False)
                connection.execute(text(f"ALTER TABLE {db_entity} "
                                        f"ALTER COLUMN {self.primary_keys[entity]} BIGINT NOT NULL"))
                connection.execute(text(
                    f"ALTER TABLE {db_entity} ADD CONSTRAINT PK_{entity} PRIMARY KEY ({self.primary_keys[entity]});"))

            for parent_table in self.relations:
                db_parent_table = f"{database_name}.{parent_table}"
                for child_table in self.relations[parent_table]:
                    db_child_table = f"{database_name}.{child_table}"
                    connection.execute(
                        text(f"ALTER TABLE {db_child_table} ADD CONSTRAINT FK_{child_table}_{parent_table} "
                             f"FOREIGN KEY ({self.relations[parent_table][child_table][1]}) "
                             f"REFERENCES {db_parent_table}({self.relations[parent_table][child_table][0]});"))
