from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("Create DataFrame").getOrCreate()

# Create a list of data
data = [("Aditya", 25), ("Chirag", 30), ("Aryan", 35)]

# Create a DataFrame from the data
df = spark.createDataFrame(data, ["Name", "Age"])

# Show the DataFrame
df.show()
