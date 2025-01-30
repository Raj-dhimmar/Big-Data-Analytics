from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, avg, desc

# Start Spark session
spark = SparkSession.builder.appName("RetailStoreAnalysis").getOrCreate()

# File path in Google Cloud Storage
file_path = "gs://assignment-bigdata-bucket/retailstore.csv"

# Load the CSV file(Dataset
df = spark.read.option("header", "true").csv(file_path)

# Convert necessary columns to appropriate data types
df = df.withColumn("Salary", col("Salary").cast("float"))
df = df.withColumn("Age", col("Age").cast("int"))
df = df.withColumn("CustomerID", col("CustomerID").cast("int"))

# 1. Average Salary by Country
print("Average Salary by Country:")
df_grouped_by_country = df.groupBy("Country").agg(
    avg("Salary").alias("average_salary")
).orderBy(desc("average_salary"))
df_grouped_by_country.show()

# Save the result of Average Salary by Country to my bigdata bucket
df_grouped_by_country.write.csv("gs://assignment-bigdata-bucket/output/average_salary_by_country", header=True)

# 2. Gender Distribution
print("\nGender Distribution:")
df_grouped_by_gender = df.groupBy("Gender").agg(
    count("CustomerID").alias("count")
).orderBy(desc("count"))
df_grouped_by_gender.show()

# Save the result of Gender Distribution to my bigdata bucket
df_grouped_by_gender.write.csv("gs://assignment-bigdata-bucket/output/gender_distribution", header=True)

# 3. Average Age of Customers
print("\nAverage Age of Customers:")
average_age = df.agg(avg("Age").alias("average_age")).collect()
print(f"Average Age: {average_age[0]['average_age']}")

# Save the result of Average Age to my bigdata bucket
average_age_df = spark.createDataFrame([(average_age[0]['average_age'],)], ["average_age"])
average_age_df.write.csv("gs://assignment-bigdata-bucket/output/average_age.csv", header=True)

# 4. Total Customers by Country
print("\nTotal Customers by Country:")
df_grouped_by_country_customers = df.groupBy("Country").agg(
    count("CustomerID").alias("total_customers")
).orderBy(desc("total_customers"))
df_grouped_by_country_customers.show()

# Save the result of Total Customers by Country to my bigdata bucket
df_grouped_by_country_customers.write.csv("gs://assignment-bigdata-bucket/output/total_customers_by_country", header=True)

# 5. Top 5 Countries by Total Salary Spend
print("\nTop 5 Countries by Total Salary Spend:")
df_grouped_by_country_salary = df.groupBy("Country").agg(
    sum("Salary").alias("total_salary_spent")
).orderBy(desc("total_salary_spent"))
df_grouped_by_country_salary.show(5)

# Save the result of Top 5 Countries by Total Salary Spend to my bigdata bucket
df_grouped_by_country_salary.write.csv("gs://assignment-bigdata-bucket/output/top_5_countries_by_salary_spend", header=True)

# Stop Spark session
spark.stop()
