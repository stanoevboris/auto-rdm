from logger_config import logger
from typing import Optional

from sqlalchemy import engine
from conf import *
from abc import ABC, abstractmethod
import pandas as pd
from sqlalchemy import create_engine, text

from sqlalchemy_utils import database_exists, create_database
import sqlalchemy as sa


def create_connection(database_name: str):
    mssql_db = MSSQLDatabase(database="woe", target_schema=database_name)
    mssql_engine = create_engine(mssql_db.connection_url, echo=True)
    if not database_exists(mssql_engine.url):
        create_database(mssql_engine.url)

    return mssql_engine


class Database(ABC):
    def __init__(self, target_schema: str, database: Optional[str] = None, include_all_schemas: bool = False):
        self.database = database
        self.target_schema = target_schema
        self.include_all_schemas = include_all_schemas

    @abstractmethod
    def get_connection_url(self):
        pass

    @abstractmethod
    def get_table(self, schema: Optional[str], table_name: str, connection):
        pass

    @abstractmethod
    def get_primary_keys(self):
        pass

    @abstractmethod
    def get_foreign_keys(self):
        pass

    def get_data(self):
        db_engine = create_engine(self.get_connection_url(), echo=True)
        tables_dict = {}
        with db_engine.connect() as connection:
            schemas = [self.target_schema] if not self.include_all_schemas else sa.inspect(db_engine).get_schema_names()

            for schema in schemas:
                tables = sa.inspect(db_engine).get_table_names(schema=schema)
                for table in tables:
                    tables_dict[table] = self.get_table(schema=schema, table_name=table, connection=connection)

            logger.info(f"Total tables read: {len(tables_dict)}")
            logger.info(f"Tables read: {list(tables_dict.keys())}")

            pks = pd.read_sql(text(self.get_primary_keys()), connection)
            fks = pd.read_sql(text(self.get_foreign_keys()), connection)

        pks_dict = dict(zip(pks['TableName'], pks['PrimaryKeyColumn']))
        fk_graph = fks[['ChildTable', 'ChildColumn', 'ReferencedTable', 'ReferencedColumn']].values.tolist()
        return tables_dict, pks_dict, fk_graph


class MSSQLDatabase(Database):

    def __init__(self, target_schema: str, database: Optional[str] = None, include_all_schemas: bool = False):
        super().__init__(target_schema, database, include_all_schemas)
        self.username = MSSQL_READ_USER
        self.password = MSSQL_READ_PASS
        self.host = MSSQL_HOST
        self.port = MSSQL_PORT
        self.query = MSSQL_QUERY

        self.database = database
        self.target_schema = target_schema
        self.include_all_schemas = include_all_schemas

        self.connection_url = self.get_connection_url()

    def get_connection_url(self):
        return engine.URL.create(
            MSSQL_DBAPI,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query=self.query
        )

    def get_table(self, schema: Optional[str], table_name: str, connection):
        if schema:
            df = pd.read_sql(text(f"SELECT * FROM {schema}.{table_name}"), connection)
        else:
            df = pd.read_sql(text(f"SELECT * FROM {self.target_schema}.{table_name}"), connection)
        return df

    def get_primary_keys(self):
        query = f"""SELECT
                            t.name AS TableName,
                            c.name AS PrimaryKeyColumn
                        FROM
                            sys.tables AS t
                        INNER JOIN
                            sys.indexes AS i
                        ON
                            t.object_id = i.object_id
                        INNER JOIN
                            sys.index_columns AS ic
                        ON
                            i.object_id = ic.object_id
                            AND i.index_id = ic.index_id
                        INNER JOIN
                            sys.columns AS c
                        ON
                            ic.object_id = c.object_id
                            AND ic.column_id = c.column_id
                        WHERE
                            i.is_primary_key = 1"""

        if not self.include_all_schemas:
            schema_filter = f" AND t.schema_id = SCHEMA_ID('{self.target_schema}')"
            query += schema_filter

        return query + ";"

    def get_foreign_keys(self):
        query = f""" SELECT   
                        fk.name AS ForeignKeyName,
                        OBJECT_NAME(fkc.parent_object_id) AS ChildTable,
                        COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ChildColumn,
                        OBJECT_NAME(fkc.referenced_object_id) AS ReferencedTable,
                        COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
                    FROM
                        sys.foreign_keys AS fk
                    INNER JOIN
                        sys.foreign_key_columns AS fkc
                    ON
                        fk.object_id = fkc.constraint_object_id
                    WHERE
                        1 = 1"""

        if not self.include_all_schemas:
            schema_filter = f" AND OBJECT_SCHEMA_NAME(fkc.parent_object_id) = '{self.target_schema}'" \
                            f"AND OBJECT_SCHEMA_NAME(fkc.referenced_object_id) = '{self.target_schema}'"""
            query += schema_filter

        return query + ";"


class MYSQLDatabase(Database):

    def __init__(self, target_schema: str):
        super().__init__(target_schema)
        self.username = MYSQL_USER
        self.password = MYSQL_PASS
        self.host = MYSQL_HOST
        self.port = MYSQL_PORT
        self.database = target_schema
        self.query = MSSQL_QUERY

        self.target_schema = target_schema

        if not self.database:
            raise ValueError("Database missing!")
        self.connection_url = self.get_connection_url()

    def get_connection_url(self):
        return engine.URL.create(
            MYSQL_DBAPI,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )

    def get_table(self, schema: Optional[str], table_name: str, connection):
        df = pd.read_sql(text(f"SELECT * FROM {self.target_schema}.{table_name}"), connection)
        return df

    def get_primary_keys(self):
        return f"""SELECT
                        TABLE_NAME as TableName,
                         COLUMN_NAME as PrimaryKeyColumn
                    FROM 
                        INFORMATION_SCHEMA.COLUMNS 
                    WHERE 
                        TABLE_SCHEMA = '{self.target_schema}' 
                        AND COLUMN_KEY = 'PRI' 
                    ORDER BY TABLE_NAME;"""

    def get_foreign_keys(self):
        return f"""SELECT
                        CONSTRAINT_NAME AS ForeignKeyName,
                        TABLE_NAME AS ChildTable,
                        COLUMN_NAME AS ChildColumn,
                        REFERENCED_TABLE_NAME AS ReferencedTable,
                        REFERENCED_COLUMN_NAME AS ReferencedColumn
                    FROM
                        information_schema.KEY_COLUMN_USAGE
                    WHERE
                        CONSTRAINT_SCHEMA = '{self.target_schema}'
                        AND REFERENCED_TABLE_NAME IS NOT NULL;"""


def get_database(sql_type: str, target_schema: str, database: Optional[str] = None,
                 include_all_schemas: bool = False) -> Database:
    database_classes = {
        "mssql": MSSQLDatabase,
        "mysql": MYSQLDatabase
    }

    database_class = database_classes.get(sql_type)
    if not database_class:
        raise ValueError(f"Unsupported database type: {sql_type}")

    # Prepare keyword arguments according to the class requirements
    kwargs = {"target_schema": target_schema}
    if sql_type == "mssql":
        kwargs["database"] = database
        kwargs["include_all_schemas"] = include_all_schemas

    # Instantiate the database class with appropriate kwargs
    return database_class(**kwargs)
