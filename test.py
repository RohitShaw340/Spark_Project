from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("Read HDFS Text File").getOrCreate()

# Read the text file from HDFS
text_file = spark.read.text("hdfs://localhost:9864/hadoop_installation_guide")

# Print the content of the text file
text_file.show(truncate=False)

# Stop the SparkSession
spark.stop()
