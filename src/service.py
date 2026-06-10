from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType

from config import AppConfig
from storage import SqlServerStorage


class FoodClusterService:
    def __init__(self, config: AppConfig):
        self.config = config

    def _create_spark(self) -> SparkSession:
        s = self.config.spark
        spark = (
            SparkSession.builder.appName(s.app_name)
            .master(s.master)
            .config("spark.driver.memory", s.driver_memory)
            .config("spark.sql.shuffle.partitions", s.shuffle_partitions)
            .config("spark.jars.packages", "com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel(s.log_level)
        return spark

    def fit(self):
        spark = self._create_spark()
        storage = SqlServerStorage(self.config.sqlserver)
        try:
            path = self.config.data.input_path
            raw = spark.read.parquet(path)
            numeric_cols = [
                f.name
                for f in raw.schema.fields
                if isinstance(f.dataType, NumericType)
            ]
            source = raw.select(
                *[F.col(c).cast("double").alias(c) for c in numeric_cols]
            )

            storage.ensure_database()
            storage.write(source, self.config.sqlserver.input_table)
            df = storage.read(spark, self.config.sqlserver.input_table)

            imputed_cols = [f"{c}_imp" for c in numeric_cols]
            pipeline = Pipeline(
                stages=[
                    Imputer(
                        inputCols=numeric_cols,
                        outputCols=imputed_cols,
                        strategy=self.config.preprocessing.imputer_strategy,
                    ),
                    VectorAssembler(inputCols=imputed_cols, outputCol="features_raw"),
                    StandardScaler(
                        inputCol="features_raw",
                        outputCol="features",
                        withMean=True,
                        withStd=True,
                    ),
                    KMeans(
                        featuresCol="features",
                        predictionCol="prediction",
                        k=self.config.training.k,
                        seed=self.config.training.seed,
                    ),
                ]
            )
            predictions = pipeline.fit(df).transform(df)

            score = ClusteringEvaluator(
                metricName=self.config.training.metric_name,
                distanceMeasure=self.config.training.distance_measure,
            ).evaluate(predictions)
            print(f"fit. {self.config.training.metric_name}={score:.4f}")

            storage.write(
                predictions.select(*numeric_cols, "prediction"),
                self.config.sqlserver.predictions_table,
            )
            storage.write(
                spark.createDataFrame(
                    [(self.config.training.metric_name, float(score))],
                    ["metric", "value"],
                ),
                self.config.sqlserver.metrics_table,
            )
        finally:
            spark.stop()
