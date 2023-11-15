# For Pyspark with No none
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType
from termcolor import colored
from tqdm import tqdm
import numpy as np
import pandas as pd
from IPython.display import display

def ReduceMemory(df: pd.DataFrame):
    """
    This function reduces the associated dataframe's memory usage.
    It reassigns the data-types of columns according to their min-max values.
    It also displays the dataframe information after memory reduction.
    """;
    
    # Reducing float column memory usage:-
    for col in tqdm(df.iloc[0:2, 1:].select_dtypes('float').columns):
        col_min = np.amin(df[col].dropna());
        col_max = np.amax(df[col].dropna());
        
        if col_min >= np.finfo(np.float16).min and col_max <= np.finfo(np.float16).max: 
            df[col] = df[col].astype(np.float16)
        elif col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max : 
            df[col] = df[col].astype(np.float32)
        else: pass;

    # Reducing integer column memory usage:-
    for col in tqdm(df.iloc[0:2, 1:].select_dtypes('int').columns):
        col_min = df[col].min(); 
        col_max = df[col].max();
        
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8);
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16);
        elif col_min >= np.iinfo(np.int32).min & col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32);
        else: pass;
        
    print(colored(f"\nDataframe information after memory reduction\n", 
                  color = 'blue', attrs= ['bold']));
    display(df.info()); 
    
    return df;
    

    