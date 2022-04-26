from pyspark.sql import SparkSession

MAX_MEMORY="5g"
spark = SparkSession.builder.appName("taxi-fare-prediciton")\
                .config("spark.executor.memory", MAX_MEMORY)\
                .config("spark.driver.memory", MAX_MEMORY)\
                .getOrCreate()

hdfs_namenode = "hadoop-namenode:8020"

trips_df = spark.read.csv(f"hdfs://{hdfs_namenode}/data/raw/tripdata/yellow_tripdata_2021-01.csv", header=True, inferSchema=True)

trips_df.createOrReplaceTempView("trips")
query = """
SELECT 
    passenger_count,
    PULocationID as pickup_location_id,
    DOLocationID as dropoff_location_id,
    trip_distance,
    HOUR(tpep_pickup_datetime) as pickup_time,
    DATE_FORMAT(TO_DATE(tpep_pickup_datetime), 'EEEE') AS day_of_week,
    total_amount
FROM
    trips
WHERE
    total_amount < 5000
    AND total_amount > 0
    AND trip_distance > 0
    AND trip_distance < 500
    AND passenger_count < 4
    AND TO_DATE(tpep_pickup_datetime) >= '2021-01-01'
    AND TO_DATE(tpep_pickup_datetime) < '2021-08-01'
"""
data_df = spark.sql(query)

train_df, test_df = data_df.randomSplit([0.8, 0.2], seed=1)

train_df.write.format("parquet").mode('overwrite').save(f"hdfs://{hdfs_namenode}/data/parquet/tripdata/train/")
test_df.write.format("parquet").mode('overwrite').save(f"hdfs://{hdfs_namenode}/data/parquet/tripdata/test/")
