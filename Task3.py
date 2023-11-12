from utils import reduce_memory as ReduceMemory

# Create argparser
from pyspark.ml.feature import (
    Imputer,
    StandardScaler,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml import Pipeline, Transformer
import os
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit, monotonically_increasing_id
from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
import pyspark
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

'''parser = argparse.ArgumentParser(
    description="Spark Machine Learning Pipeline for FIFA Dataset"
)

parser.add_argument("--input_path", type=str,
                    help="Input path", default="/fifadata")
parser.add_argument("--output_path", type=str,
                    help="Output path", default="/output")
parser.add_argument("--use_clean_data", type=bool, help="Use clean data or not", default=False)
parser.add_argument("--verbose", type=int, help="Verbose or not", default=1)

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
use_clean_data = args.use_clean_data
'''

# import related libraries

# import spark libraries

# init a spark session
spark = SparkSession.builder.appName("FIFA Dataset").getOrCreate()

# define some config
continuous_cols = [
    "potential",
    "value_eur",
    "wage_eur",
    "age",
    "height_cm",
    "weight_kg",
    "club_team_id",
    "league_level",
    "nationality_id",
    "weak_foot",
    "skill_moves",
    "international_reputation",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
    "ls",
    "st",
    "rs",
    "lw",
    "lf",
    "cf",
    "rf",
    "rw",
    "lam",
    "cam",
    "ram",
    "lm",
    "lcm",
    "cm",
    "rcm",
    "rm",
    "lwb",
    "ldm",
    "cdm",
    "rdm",
    "rwb",
    "lb",
    "lcb",
    "cb",
    "rcb",
    "rb",
    "gk",
]
position_binary_cols = [
    "Position_CF",
    "Position_LW",
    "Position_LM",
    "Position_RM",
    "Position_RW",
    "Position_ST",
    "Position_GK",
    "Position_CM",
    "Position_CDM",
    "Position_RB",
    "Position_CB",
    "Position_CAM",
    "Position_LB",
    "Position_RWB",
    "Position_LWB",
]
nominal_cols = ["league_name", "club_position", "work_rate"]
ordinal_cols = ["year", "preferred_foot"]

cols_to_drop = [
    'long_name',
    "player_url",
    "player_face_url",
    "club_logo_url",
    "club_flag_url",
    "nation_logo_url",
    "nation_flag_url",
    "sofifa_id",
    "short_name",
    "dob",
    "club_name",
    "club_jersey_number",
    "club_loaned_from",
    "nationality_name",
    "nation_jersey_number",
    "body_type",
    "real_face",
    "goalkeeping_speed",
    "club_contract_valid_until",
    "nation_team_id",
    "nation_position",
    "player_tags",
    "player_traits",
    "release_clause_eur",
    "long_name",
]


# Create a class to process data
# import spark ml related libraries
class OutcomeCreater(Transformer):
    def __init__(self):
        super().__init__()
        
    def _transform(self, df):
        df = df.withColumnRenamed(
            "overall", "outcome"
        )  # rename the overall column to outcome
        return df


class ColumnDropper(Transformer):
    def __init__(self, cols_to_drop=None):
        super().__init__()
        self.cols_to_drop = cols_to_drop

    def _transform(self, df):
        return df.drop(*self.cols_to_drop)


class DataPreprocess1(Transformer):
    """for columns like ls, st..., gk
    columns that contains + or - as string
    """

    def __init__(self, cols_to_preprocess) -> None:
        super().__init__()
        self.cols_to_preprocess = cols_to_preprocess

    def _transform(self, df):
        from pyspark.sql.functions import split
        from pyspark.sql.types import IntegerType

        for col in self.cols_to_preprocess:
            df = df.withColumn(col, split(
                df[col], r'\+|-').getItem(0).cast(IntegerType()))
        return df


class DataPreprocess2(Transformer):
    """
    Transforme the columns in Positions to binary columns
    """

    def __init__(self, cols_to_preprocess) -> None:
        super().__init__()
        self.cols_to_preprocess = cols_to_preprocess

    def _transform(self, dataset: DataFrame) -> DataFrame:
        from pyspark.sql.functions import split, when, col, array_contains
        import itertools
        split_positions = split(dataset['self.cols_to_preprocess'], ', ')
        self.distinct_positions = list(set(list(itertools.chain(
            *dataset.select(split_positions.alias('positions')).distinct().rdd.flatMap(lambda x: x).collect()))))
        for position in self.distinct_positions:
            dataset = dataset.withColumn(
                'Position_' + position,
                when(array_contains(split_positions, position), 1).otherwise(0)
            ).drop(self.cols_to_preprocess)
        return dataset
    
class MissingValueImputer(Transformer):
    def __init__(self, cols_to_impute=None):
        super().__init__()
        self.cols_to_impute = cols_to_impute

    def _transform(self, df):
        raise NotImplementedError

def get_preprocess_pipeline():

    # Stage for columns to preprocess2
    stage_column_pre1 = DataPreprocess1(["ls",
                                        "st",
                                        "rs",
                                        "lw",
                                        "lf",
                                        "cf",
                                        "rf",
                                        "rw",
                                        "lam",
                                        "cam",
                                        "ram",
                                        "lm",
                                        "lcm",
                                        "cm",
                                        "rcm",
                                        "rm",
                                        "lwb",
                                        "ldm",
                                        "cdm",
                                        "rdm",
                                        "rwb",
                                        "lb",
                                        "lcb",
                                        "cb",
                                        "rcb",
                                        "rb",
                                        "gk",])
    
    # Stage for columns to preprocess2
    stage_column_pre2 = DataPreprocess2(["player_positions"])
    # Stage where nominal columns are transformed  to index columns using StringIndexer
    nominal_id_cols = [x+"_index" for x in nominal_cols]
    nominal_onehot_cols = [x+"_onehot" for x in nominal_cols]
    stage_nominal_indexer = StringIndexer(
        inputCols=nominal_cols, outputCols=nominal_id_cols)

    # Stage where ordinal columns are transformed to index columns using StringIndexer
    ordinal_id_cols = [x+"_index" for x in ordinal_cols]
    stage_ordinal_indexer = StringIndexer(
        inputCols=ordinal_cols, outputCols=ordinal_id_cols)

    feature_cols = continuous_cols + position_binary_cols + \
        ordinal_id_cols + nominal_onehot_cols
    # Stage where all the features are assembled into a single vector
    stage_vector_assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="vectorized_features")

    # Stage where we scale the columns
    stage_scaler = StandardScaler(
        inputCol='vectorized_features', outputCol='features')

    # Stage for creating the outcome column representing whether there is attack
    stage_outcome = OutcomeCreater()
    
    # Stage for columns dropping
    stage_column_dropper = ColumnDropper(cols_to_drop=cols_to_drop + feature_cols + ['vectorized_features'])
    
    # Connect the columns into a pipeline
    pipeline = Pipeline(stages=[stage_column_pre1,
                                stage_column_pre2,
                                stage_nominal_indexer,
                                stage_ordinal_indexer,
                                stage_vector_assembler,
                                stage_scaler,
                                stage_outcome,
                                stage_column_dropper])
    
    return pipeline


def load_data(input_path, output_path):
    full_data_path = os.path.join(output_path, "full_data.csv")

    if not os.path.exists(full_data_path):
        data_path = os.path.join(input_path, 'fifadata')
        if os.path.exists(data_path):
            print("Data folder exists")
        else:
            print("Data folder does not exist")
            os.makedirs(data_path)
            print("Sussessfully created data folder")

        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        print(csv_files)
        combined_df = None
        for file in csv_files:
            year = file.split("players_")[1].split(".csv")[0]
            df = spark.read.csv(file, header=True, inferSchema=True)
            df = df.withColumn("year", lit(year)) # this is the unique column 'year'
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.union(df)
        combined_df = combined_df.withColumn("id", monotonically_increasing_id())

        # Write the concatenated DataFrame to a new CSV file
        output_file = "/Users/dylan/DylanLi/Code_Repo/CMU18763_Projects1/full_data.csv"
        ReduceMemory(combined_df.toPandas()).to_csv(output_file)
    else: 
        df = spark.read.csv(full_data_path, header=True, inferSchema=True)
