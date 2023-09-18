# import libariies
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SQLContext
import os
import pandas as pd 


def get_csv_file_paths(directory):
  csv_file_paths = []
  for root, directories, files in os.walk(directory):
    for file in files:
      if os.path.isfile(os.path.join(root, file)):
        file_name, file_extension = os.path.splitext(file)
        if file_extension == '.csv':
          csv_file_paths.append(os.path.join(root, file))
  return csv_file_paths

def add_year(file_path): 
    df = pd.read_csv(file_path)
    # output final year
    
    df['year'] = int(f'20{file_path[-6:-4]}')
    df.to_csv(file_path, index=False)

file_paths = get_csv_file_paths('/home/dylan/dylan_repo/CMU18763_Projects1/fifadata')
#connect to local postgreSQL database using SparkSQL 
# config for Spark
master = 'local'
conf = pyspark.SparkConf().set("spark.driver.host", '127.0.0.1')

# Create Spark context
sc = pyspark.SparkContext(master=master, appName="myAppName", conf=conf)

# Create SQL context
sqlContext = SQLContext(sc)

# Create Spark session using SQL context
spark = sqlContext.sparkSession.builder.getOrCreate()

# add a new columns named year using add_year function
for file_path in file_paths:
    add_year(file_path)

# read multiple csv files into spark dataframe
df = spark.read.csv(file_paths, header=True, inferSchema=True)

# Insert unique id
from pyspark.sql.functions import monotonically_increasing_id
df = df.withColumn("id", monotonically_increasing_id())

# Write to A Table 
# Here we use a managed table, because 
# we want to be able to query it with Spark SQL
df.write.mode("overwrite").saveAsTable("fifa")



