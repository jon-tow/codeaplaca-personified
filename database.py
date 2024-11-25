import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple, Type, TypeVar

import duckdb
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Table(BaseModel):
    id: str

    @classmethod
    def primary_key(cls) -> Tuple[str]:
        return ("id TEXT",)


class ChatTurn(Table):
    system: str
    user: str
    assistant: str = "None"
    model: str
    language: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


T = TypeVar("T", bound=Table)


class Database:
    """A class to manage database operations using DuckDB."""

    def __init__(self, db_name: str = ":memory:"):
        self.conn = duckdb.connect(database=db_name)

    def create_table(
        self,
        model: Type[T],
        table_name: str,
        index_columns: List[str] | None = None,
    ):
        """
        Create a new table in the database based on a Pydantic `Table`-derived model.

        Args:
            model (Type[T]): A Pydantic model class defining the table structure.
            table_name (str): The name of the table to be created.
            index_columns (List[str] | None): Columns to be indexed for faster querying.

        Raises:
            Exception: If there's an error during table creation.
        """
        try:
            fields = model.__annotations__.items()
            # Remove `id` field from the list of fields as it's a primary key
            fields = [
                (field_name, field_type)
                for field_name, field_type in fields
                if field_name != "id"
            ]
            columns = ", ".join(
                f"{field_name} {self._pydantic_to_sql_type(field_type)}"
                for field_name, field_type in fields
            )
            primary_key_fields = ", ".join(model.primary_key())
            if hasattr(model, "unique_constraint"):
                unique_constraint_fields = ", ".join(model.unique_constraint())
                create_table_query = (
                    f"CREATE TABLE IF NOT EXISTS {table_name} "
                    f"({primary_key_fields} PRIMARY KEY, {columns}, "
                    f"CONSTRAINT unique_{table_name} UNIQUE ({unique_constraint_fields}))"
                )
            else:
                create_table_query = (
                    f"CREATE TABLE IF NOT EXISTS {table_name} "
                    f"({primary_key_fields} PRIMARY KEY, {columns})"
                )
            self.conn.execute(create_table_query)

            if index_columns:
                for column in index_columns:
                    self.conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column} "
                        f"ON {table_name}({column})"
                    )
            logger.info(
                f"Table {table_name} created successfully with indexes on "
                f"{index_columns}"
            )
        except Exception as e:
            logger.error(f"Error creating table `{table_name}`: {e}")
            raise

    def _pydantic_to_sql_type(self, pydantic_type: Any) -> str:
        if pydantic_type == str or pydantic_type == Enum:
            return "TEXT"
        elif pydantic_type == int:
            return "INTEGER"
        elif pydantic_type == float:
            return "DOUBLE"
        elif pydantic_type == bool:
            return "BOOLEAN"
        elif pydantic_type == datetime:
            return "TIMESTAMP"
        else:
            raise ValueError(f"Unsupported Pydantic type: {pydantic_type}")

    def add(
        self, table_name: str, items: List[BaseModel] | BaseModel
    ) -> List[BaseModel]:
        """Adds items into the table and returns the items that were successfully added."""

        def escape_single_quotes(value: Any) -> Any:
            """Escapes single quotes in a string to be used in a SQL query."""
            if isinstance(value, str):
                return value.replace("'", "''")
            return value

        try:
            if not isinstance(items, list):
                items = [items]
            columns = items[0].model_dump().keys()
            # Manual bulk insertion b/c DuckDB `executemany` fails to insert values
            # when `RETURNING *` is used.
            values = ", ".join(
                [
                    "("
                    + ", ".join(
                        [f"'{escape_single_quotes(getattr(item, c))}'" for c in columns]
                    )
                    + ")"
                    for item in items
                ]
            ).strip()
            query = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns).strip()}) VALUES {values} RETURNING *;"
            results = self.sql(query)
            return results
        except Exception as e:
            self.conn.rollback()  # Rollback if anything goes wrong
            logger.error(f"Transaction failed - rolling back:\n```\n{e}\n```")
            raise

    def sample(
        self,
        table_name: str,
        k: int = 1,
        condition: str | None = None,
        # seed: int = 123456789, TODO: `duckdb.random` does not support seeds!?
    ) -> List[Dict[str, Any]]:
        """Adapted from: https://gist.github.com/alecco/9976dab8fda8256ed403054ed0a65d7b
        Terribly inefficient for large tables, but good enough for small ones...
        """
        try:
            query = f"""
            SELECT * FROM {table_name}
            WHERE rowid IN (
                SELECT rowid FROM {table_name}
            """
            if condition:
                query += f" WHERE {condition}"
            query += f"""
                ORDER BY random()
                LIMIT {k}
            )
            """
            result = self.conn.execute(query).fetchall()
            columns = [desc[0] for desc in self.conn.description]  # type: ignore
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Error sampling from `{table_name}`: {e}")
            raise

    def get(
        self, table_name: str, condition: str | None = None
    ) -> List[Dict[str, Any]]:
        try:
            query = f"SELECT * FROM {table_name} "
            if condition:
                query += f"WHERE {condition}"
            result = self.conn.execute(query).fetchall()
            columns = [desc[0] for desc in self.conn.description]  # type: ignore
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            raise

    def count(self, table_name: str) -> int:
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            result = self.conn.execute(query).fetchone()
            count = result[0] if result else 0
            # logger.info(f"Table {table_name} has {count} rows")
            return count
        except Exception as e:
            logger.error(f"Error getting row count for table {table_name}: {e}")
            raise

    def write_to_parquet(self, table_name: str, file_path: str):
        try:
            table = self.conn.execute(f"SELECT * FROM {table_name}").arrow()
            pq.write_table(table, file_path)
            logger.info(f"Written table {table_name} to {file_path}")
        except Exception as e:
            logger.error(
                f"Error writing table {table_name} to Parquet file {file_path}: {e}"
            )
            raise

    def write_to_jsonl(self, table_name: str, file_path: str):
        import json
        from datetime import datetime

        try:
            query = f"SELECT * FROM {table_name}"
            rows = self.conn.execute(query).fetchall()
            columns = [desc[0] for desc in self.conn.description]

            # Custom JSON encoder to handle datetime and UUID
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    if isinstance(obj, uuid.UUID):
                        return str(obj)
                    return super().default(obj)

            with open(file_path, "w") as f:
                for row in rows:
                    json.dump(dict(zip(columns, row)), f, cls=CustomJSONEncoder)
                    f.write("\n")

            logger.info(f"Written table {table_name} to JSONL file {file_path}")
        except Exception as e:
            logger.error(
                f"Error writing table {table_name} to JSONL file {file_path}: {e}"
            )
            raise

    def sql(self, query: str, args: Tuple = ()) -> List[Dict[str, Any]]:
        try:
            result = self.conn.execute(query, args).fetchall()
            columns = [desc[0] for desc in self.conn.description]  # type: ignore
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
