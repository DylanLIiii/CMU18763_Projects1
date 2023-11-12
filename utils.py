# For Pyspark with No none
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType
from termcolor import colored
from tqdm import tqdm
import numpy as np

def reduce_memory(df):
    """
    This function reduces the associated DataFrame's memory usage.
    It reassigns the data-types of columns according to their min-max values.
    It also displays the DataFrame information after memory reduction.
    """
    
    # Reducing float column memory usage:-
    for col_name, data_type in tqdm(df.dtypes):
        if data_type == 'float':
            col_min = df.select(col(col_name)).dropna().rdd.map(lambda x: x[0]).min()
            col_max = df.select(col(col_name)).dropna().rdd.map(lambda x: x[0]).max()
            
            if col_min >= np.finfo(np.float16).min and col_max <= np.finfo(np.float16).max: 
                df = df.withColumn(col_name, col(col_name).cast(FloatType()))
            elif col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max : 
                df = df.withColumn(col_name, col(col_name).cast(FloatType()))
            else: pass

    # Reducing integer column memory usage:-
    for col_name, data_type in tqdm(df.dtypes):
        if data_type == 'int':
            col_min = df.select(col(col_name)).rdd.map(lambda x: x[0]).min()
            col_max = df.select(col(col_name)).rdd.map(lambda x: x[0]).max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
            elif col_min >= np.iinfo(np.int32).min & col_max <= np.iinfo(np.int32).max:
                df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
            else: pass
        
    print(colored(f"\nDataframe information after memory reduction\n", 
                  color = 'blue', attrs= ['bold']))
    df.printSchema()
    
    return df
    