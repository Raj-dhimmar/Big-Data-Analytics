import pandas as pd
from collections import defaultdict

# Load the dataset into my bigdata bucket
file_path = "gs://assignment-bigdata-bucket/retailstore.csv"
data = pd.read_csv(file_path)

# Mapper function
def mapper(row):
    """
    Processes a single row of the dataset and emits a key-value pair.
    Key: Country
    Value: 1 (represents one customer)
    """
    country = row['Country']
    return (country, 1)

# Reducer function
def reducer(key, values):
    """
    Aggregates values for a given key.
    Key: Country
    Values: List of integers (e.g., [1, 1, 1])
    """
    return (key, sum(values))

# Simulate the MapReduce job
def mapreduce_job(data):
    """
    Runs the MapReduce job on the given dataset.
    """
    # Map phase
    intermediate_results = []
    for _, row in data.iterrows():
        intermediate_results.append(mapper(row))
    
    # Shuffle phase
    grouped_data = defaultdict(list)
    for key, value in intermediate_results:
        grouped_data[key].append(value)
    
    # Reduce phase
    final_results = []
    for key, values in grouped_data.items():
        final_results.append(reducer(key, values))
    
    return final_results

# Run the MapReduce job
results = mapreduce_job(data)

# Results
for country, count in results:
    print(f"{country}: {count}")
