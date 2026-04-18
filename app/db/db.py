import os
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

load_dotenv()


class Database:
    _pool = None

    @classmethod
    def initialize(cls):
        if cls._pool is None:
            cls._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,  # adjust based on load
                dsn=os.getenv("DATABASE_URL"),
            )

    @classmethod
    def get_connection(cls):
        if cls._pool is None:
            raise Exception("Connection pool not initialized")
        return cls._pool.getconn()

    @classmethod
    def return_connection(cls, conn):
        if cls._pool:
            cls._pool.putconn(conn)

    @classmethod
    def close_all(cls):
        if cls._pool:
            cls._pool.closeall()