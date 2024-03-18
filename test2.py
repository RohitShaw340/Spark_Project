import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

spark = SparkSession.builder.master("local").appName("hdfs_test").getOrCreate()

booksdata = spark.read.csv("hdfs://localhost:9820/spark_project_dataset/mail_data3")
booksdata.show(5)
