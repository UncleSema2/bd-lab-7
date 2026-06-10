import pymssql
from pyspark.sql import DataFrame, SparkSession

from config import SqlServerConfig

JDBC_DRIVER = "com.microsoft.sqlserver.jdbc.SQLServerDriver"


class SqlServerStorage:
    def __init__(self, config: SqlServerConfig):
        self.config = config

    @property
    def url(self) -> str:
        return (
            f"jdbc:sqlserver://{self.config.host}:{self.config.port};"
            f"databaseName={self.config.database};encrypt=false"
        )

    @property
    def properties(self) -> dict:
        return {
            "user": self.config.user,
            "password": self.config.password,
            "driver": JDBC_DRIVER,
        }

    def ensure_database(self) -> None:
        conn = pymssql.connect(
            server=self.config.host,
            port=str(self.config.port),
            user=self.config.user,
            password=self.config.password,
            autocommit=True,
        )
        try:
            conn.cursor().execute(
                f"IF DB_ID('{self.config.database}') IS NULL "
                f"CREATE DATABASE [{self.config.database}]"
            )
        finally:
            conn.close()

    def read(self, spark: SparkSession, table: str) -> DataFrame:
        return spark.read.jdbc(self.url, table, properties=self.properties)

    def write(self, df: DataFrame, table: str, mode: str = "overwrite") -> None:
        df.write.jdbc(self.url, table, mode=mode, properties=self.properties)
