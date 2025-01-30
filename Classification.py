from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Spark Spark session
spark = SparkSession.builder \
    .appName("RetailStoreClassification") \
    .getOrCreate()

# Load the dataset into my bigdata bucket
file_path = "gs://assignment-bigdata-bucket/retailstore.csv"
spark_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Delete null values to clear the dataset
spark_df = spark_df.dropna()

# Encode categorical columns
country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="Label")

# Apply the indexers
spark_df = country_indexer.fit(spark_df).transform(spark_df)
spark_df = gender_indexer.fit(spark_df).transform(spark_df)

# Combine features into a single vector
feature_columns = ["Age", "Salary", "CountryIndex"]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="Features")
spark_df = vector_assembler.transform(spark_df)

# Scale the features
scaler = StandardScaler(inputCol="Features", outputCol="ScaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(spark_df)
spark_df = scaler_model.transform(spark_df)

# Split the data into training and testing sets
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(featuresCol="ScaledFeatures", labelCol="Label")
lr_model = lr.fit(train_data)

# Make predictions on the test set
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="Label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# Show some predictions
predictions.select("Age", "Salary", "Country", "Gender", "prediction").show(10)

# Stop the Spark session
spark.stop()
