from dataclasses import dataclass
import os
import yaml


@dataclass
class SparkConfig:
    app_name: str
    master: str
    driver_memory: str
    shuffle_partitions: str
    log_level: str


@dataclass
class DataConfig:
    input_path: str


@dataclass
class SqlServerConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    input_table: str
    predictions_table: str
    metrics_table: str


@dataclass
class PreprocessingConfig:
    imputer_strategy: str


@dataclass
class TrainingConfig:
    k: int
    seed: int
    metric_name: str
    distance_measure: str


@dataclass
class AppConfig:
    spark: SparkConfig
    data: DataConfig
    sqlserver: SqlServerConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sql = data["sqlserver"]

    return AppConfig(
        spark=SparkConfig(**data["spark"]),
        data=DataConfig(**data["data"]),
        sqlserver=SqlServerConfig(
            host=os.environ["MSSQL_HOST"],
            port=sql["port"],
            database=sql["database"],
            user=sql["user"],
            password=os.environ["MSSQL_PASSWORD"],
            input_table=sql["input_table"],
            predictions_table=sql["predictions_table"],
            metrics_table=sql["metrics_table"],
        ),
        preprocessing=PreprocessingConfig(**data["preprocessing"]),
        training=TrainingConfig(**data["training"]),
    )
