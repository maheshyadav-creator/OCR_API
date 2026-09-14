from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PGConnection

from app.core.config import get_settings


_connection_pool: pool.ThreadedConnectionPool | None = None


def initialize_connection_pool() -> None:
    """
    Create the PostgreSQL connection pool.

    This function is called during application startup, after
    PostgreSQL availability has been confirmed.
    """
    global _connection_pool

    if _connection_pool is not None:
        return

    settings = get_settings()

    _connection_pool = pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )


def close_connection_pool() -> None:
    """Close all PostgreSQL connections."""
    global _connection_pool

    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


@contextmanager
def get_db_connection() -> Generator[PGConnection, None, None]:
    """
    Get a PostgreSQL connection from the pool.

    The connection is returned to the pool automatically.
    """
    if _connection_pool is None:
        raise RuntimeError(
            "PostgreSQL connection pool has not been initialized."
        )

    connection = _connection_pool.getconn()

    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        _connection_pool.putconn(connection)


def check_database_connection() -> bool:
    """Check whether PostgreSQL is reachable."""
    settings = get_settings()

    connection = None

    try:
        connection = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=5,
        )

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()

        return result == (1,)

    except psycopg2.Error:
        return False

    finally:
        if connection is not None:
            connection.close()


def initialize_database() -> None:
    """
    Create the required PostgreSQL tables and indexes.
    """
    schema_path = Path(__file__).resolve().parent / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Database schema file not found: {schema_path}"
        )

    schema_sql = schema_path.read_text(encoding="utf-8")

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(schema_sql)

        connection.commit()