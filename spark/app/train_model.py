from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

MAX_MEMORY="5g"
spark = SparkSession.builder.appName("taxi-fare-prediciton")\
                .config("spark.executor.memory", MAX_MEMORY)\
                .config("spark.driver.memory", MAX_MEMORY)\
                .getOrCreate()

hdfs_namenode = "hadoop-namenode:8020"

train_df = spark.read.parquet(f"hdfs://{hdfs_namenode}/data/parquet/tripdata/train/")
test_df = spark.read.parquet(f"hdfs://{hdfs_namenode}/data/parquet/tripdata/train/")

vassembler = VectorAssembler(inputCols=["trip_distance"], outputCol="features")
vtrain_df = vassembler.transform(train_df)

lr = LinearRegression(
    maxIter=50,
    labelCol="total_amount",
    featuresCol="features"
)

model = lr.fit(vtrain_df)
# vtest_df = vassembler.transform(test_df)
# prediction = model.transform(vtest_df)

model.write().overwrite().save(f"hdfs://{hdfs_namenode}/data/model")
