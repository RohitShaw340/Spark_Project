from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark configuration
conf = SparkConf().setAppName("SpamClassification")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
print("Spark context created successfully.")

# Read multiple datasets from Hadoop
print("Collecting distributed data from HDFS...")
dataset_paths = [
    "hdfs://localhost:9820/spark_project_dataset/mail_data1",
    "hdfs://localhost:9820/spark_project_dataset/mail_data2",
    "hdfs://localhost:9820/spark_project_dataset/mail_data3",
]
dfs = [spark.read.csv(path, header=True, inferSchema=True) for path in dataset_paths]

print("Distributed Data collected successfully from HDFS.")


# Combine datasets into a single data source
combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = combined_df.union(df)

# Drop null rows
combined_df = combined_df.na.drop(subset=["Message"])

# Print the first 5 rows of the combined dataset
combined_df.show(5)

# Prepare data for classification
tokenizer = Tokenizer(inputCol="Message", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
label_stringIdx = StringIndexer(inputCol="Category", outputCol="labelIndex")

# Split data into train and test sets (70% train, 30% test)
train_data, test_data = combined_df.randomSplit([0.7, 0.3], seed=123)

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_stringIdx])
# Transform the data using the pipeline
transformed_df = pipeline.fit(combined_df).transform(combined_df)

# Print top 5 rows of transformed data
print("Top 5 rows of transformed data:")
transformed_df.show(5)

# Create a pipeline for the classification model
nb = NaiveBayes(featuresCol="features", labelCol="labelIndex")
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_stringIdx, nb])

# Train the model
model = pipeline.fit(train_data)

# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate model accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print("Model accuracy on test data: {:.2f}%".format(accuracy * 100))

# Stop Spark context
sc.stop()
