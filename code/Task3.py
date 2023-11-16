from code.utils import ReduceMemory
from code.train import train_model, validate_model, data_loader
from code.models import MLPRegressor, TransformerRegressor, MLPResidualRegressor

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
import torch
import torch.nn as nn
import argparse
import gc

parser = argparse.ArgumentParser(
    description="Spark Machine Learning Pipeline for FIFA Dataset"
)

parser.add_argument("input_path", type=str,
                    help="Input path", default="/fifadata")
parser.add_argument("--output_path", type=str,
                    help="Output path", default="/output")
parser.add_argument("--verbose", type=int, help="Verbose or not", default=1)
parser.add_argument("--use_wandb_log", type=int, help="Use wandb log or not", default=0)

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

SEED = 3407
IS_SPARKML = False

# seed everything for reproducibility
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed=SEED)

# init a spark session
# if linux 
os.environ['SPARK_LOCAL_IP'] = '100.112.240.153'
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
    "year"
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
# nominal_cols = ["club_position", "work_rate"]
ordinal_cols = ["preferred_foot"]

cols_to_drop = [
    'id',
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
        for column in self.cols_to_preprocess:
            split_positions = split(dataset[column], ', ')
            self.distinct_positions = list(set(list(itertools.chain(
                *dataset.select(split_positions.alias('positions')).distinct().rdd.flatMap(lambda x: x).collect()))))
            print(self.distinct_positions)
            for position in tqdm(self.distinct_positions):
                dataset = dataset.withColumn(
                    'Position_' + position,
                    when(array_contains(split_positions, position), 1).otherwise(0)
                )
            
        dataset = dataset.drop(*self.cols_to_preprocess)
        return dataset
    
class MissingValueModeImputer(Transformer):
    def __init__(self, cols_to_impute=None):
        super().__init__()
        self.cols_to_impute = cols_to_impute

    def _transform(self, df):
        if not self.cols_to_impute:
            return df
        for column_name in self.cols_to_impute:
            df = self._fill_mode(df, column_name)
        return df
    def _fill_mode(self, df, col_name):
        # Calculate the mode 
        mode = df.groupBy(col_name).count().orderBy('count', ascending=False).first()[0]
        return df.na.fill({col_name: mode})

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
    
    # Stage where nominal columns are handled by imputer
    cols_to_impute_nominal = ["league_name", "club_position"]
    stage_missing_handler = MissingValueModeImputer(cols_to_impute=cols_to_impute_nominal)
    
    # find all cols that contains missing values in continuous_cols
    cols_to_imputer_numerical = []
    for col in continuous_cols:
        if data.select(col).filter(data[col].isNull()).count() > 0:
            cols_to_imputer_numerical.append(col)
    print(cols_to_imputer_numerical)
    
    from pyspark.ml.feature import Imputer
    stage_missing_handler2 = Imputer(strategy='mean', inputCols=cols_to_imputer_numerical, outputCols=cols_to_imputer_numerical)
    
    # Stage where nominal columns are transformed  to index columns using StringIndexer
    nominal_id_cols = [x+"_index" for x in nominal_cols]
    nominal_onehot_cols = [x+"_onehot" for x in nominal_cols]
    stage_nominal_indexer = StringIndexer(
        inputCols=nominal_cols, outputCols=nominal_id_cols)
    

    
    
    # Stage where nominal columns are transformed to onehot columns using OneHotEncoder
    stage_nominal_onehot = OneHotEncoder(
        inputCols=nominal_id_cols, outputCols=nominal_onehot_cols)

    # Stage where ordinal columns are transformed to index columns using StringIndexer
    ordinal_id_cols = [x+"_index" for x in ordinal_cols]
    stage_ordinal_indexer = StringIndexer(
        inputCols=ordinal_cols, outputCols=ordinal_id_cols)

    feature_cols = continuous_cols + position_binary_cols + ordinal_id_cols + nominal_onehot_cols
    #feature_cols =  continuous_cols + ordinal_id_cols + nominal_onehot_cols

    # Stage where all the features are assembled into a single vector
    stage_vector_assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="vectorized_features")

    # Stage where we scale the columns
    stage_scaler = StandardScaler(
        inputCol='vectorized_features', outputCol='features')

    # Stage for creating the outcome column representing whether there is attack
    stage_outcome = OutcomeCreater()
    
    # Stage for columns dropping
    stage_column_dropper = ColumnDropper(cols_to_drop=cols_to_drop + feature_cols + ordinal_cols + nominal_cols + nominal_id_cols + ['vectorized_features'])
    
    # Connect the columns into a pipeline
    pipeline = Pipeline(stages=[stage_column_pre1,
                                stage_column_pre2,
                                stage_missing_handler,
                                stage_missing_handler2,
                                stage_nominal_indexer,
                                stage_nominal_onehot,
                                stage_ordinal_indexer,
                                stage_vector_assembler,
                                stage_scaler,
                                stage_outcome,
                                stage_column_dropper])
    
    return pipeline


def load_data(input_path, output_path):
    full_data_path = os.path.join(output_path, "full_data.csv")

    if not os.path.exists(full_data_path):
        os.mkdir(output_path)
        data_path = input_path
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
        df = combined_df
        # Write the concatenated DataFrame to a new CSV file
        ReduceMemory(combined_df.toPandas()).to_csv(full_data_path)
    else: 
        df = spark.read.csv(full_data_path, header=True, inferSchema=True)
    return df


data = load_data('/home/dylan/repo/CMU18763_Projects1/fifadata', '/home/dylan/repo/CMU18763_Projects1/output')
if '_c0' in data.columns:
    data = data.drop('_c0')
    
# get the pipeline 
pipeline = get_preprocess_pipeline()
pipeline_model = pipeline.fit(data)
data = pipeline_model.transform(data)

train, test = data.randomSplit([0.8, 0.2], seed=SEED)
train, val = train.randomSplit([0.75, 0.25], seed=SEED)

train.cache()
val.cache()
test.cache()

## ML Pipeline 
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import wandb

class PySparkMLModel:
    def __init__(self, model_type="logistic", learning_rate=0.01, is_wandb=False, is_plot=False):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.model = None
        self.evaluator = RegressionEvaluator(labelCol="outcome", predictionCol="prediction", metricName="rmse")
        self.wandb = is_wandb
        self.is_plot = is_plot

        # Initialize Weights & Biases
        if self.wandb: 
            wandb.init(project="pyspark_ml_model", entity="your_username")
            wandb.config.update({"learning_rate": self.learning_rate})

    def train(self, train_data):
        if self.model_type == "linear":
            self.model = LinearRegression(featuresCol="features", labelCol="outcome", regParam=self.learning_rate)
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeRegressor(featuresCol="features", labelCol="outcome")
        else:
            raise ValueError("Unsupported model type")

        model = self.model.fit(train_data)
        return model

    def evaluate(self, model, data, data_type="test"):
        predictions = model.transform(data)
        metric = self.evaluator.evaluate(predictions)
        
        print(f"RMSE on {data_type} data = {metric}")

        # Log metrics to wandb
        if self.wandb: 
            wandb.log({f"{self.model_type}_{data_type}_rmse": metric})

        # Visualizations and additional logging can be added here as needed
        predictions = model.transform(data)
        if self.is_plot:
            self.plot_residuals(predictions)
            self.log_feature_importance(model)
        
    def plot_residuals(self, predictions):
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Convert to Pandas DataFrame for easier plotting
        predictions_df = predictions.select("prediction", "outcome").toPandas()
        
        # Calculate residuals
        predictions_df['residuals'] = predictions_df['outcome'] - predictions_df['prediction']

        # Plot residuals
        plt.figure(figsize=(10, 6))
        sns.residplot(x='prediction', y='residuals', data=predictions_df, lowess=True, 
                      line_kws={'color': 'red', 'lw': 1})
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.show()
        if self.wandb:
            wandb.log({"residuals_plot": wandb.Image(plt)})
    
    def log_feature_importance(self, model):
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        if self.model_type == "decision_tree":
            # Get feature importances
            feature_importances = model.featureImportances.toArray()

            # Plot feature importances
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(range(len(feature_importances))), y=feature_importances)
            plt.title('Feature Importances')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.show()
            if self.wandb:
                # Log feature importances
                wandb.log({"feature_importances": feature_importances})
                wandb.log({"feature_importances_plot": wandb.Image(plt)})


    def save_model(self, model, model_path):
        model.write().overwrite().save(model_path)
        if self.wandb: 
            wandb.save(model_path)

    def close(self):
        wandb.finish()

    def run_pipeline(self, train, val, test):
        model = self.train(train)
        self.evaluate(model, val, "validation")
        self.evaluate(model, test, "test")
        self.save_model(model, f"{self.model_type}_model")
        if self.wandb: 
            self.close()
if IS_SPARKML:
    logistic_model = PySparkMLModel(model_type="linear", learning_rate=0.01, is_plot=False)
    logistic_model.run_pipeline(train, val, test)

    dt_model = PySparkMLModel(model_type="decision_tree", is_plot=False)
    dt_model.run_pipeline(train, val, test)

### Pytorch Pipeline 
train_loader_MLP, val_loader_MLP = data_loader(train=train, val=val, batch_size=4096)
train_loader_TS, val_loader_TS = data_loader(train=train, val=val, batch_size=256)
input_dim = train.select('features').first()[0].size
model_list = [
    MLPRegressor(input_size=input_dim, hidden_sizes=[1024, 512, 256, 64], output_size=1),
    #TransformerRegressor(input_size=input_dim, d_model=512, nhead=8, num_layers=6, output_size=1)
    MLPResidualRegressor(input_dim, [1024, 512, 256, 64], 1)
    ]

for model in model_list:
    if model.__class__.__name__ == 'MLPRegressor':
        print(f"Training {model.__class__.__name__} model, {model.__class__.__name__ == 'MLPRegressor'}")
        trained_model = train_model(model, train_loader_MLP, val_loader_MLP, epochs=300, learning_rate=0.0003)
        torch.save(trained_model.state_dict(), f"{model.__class__.__name__}.pt")
        trained_model.eval()
        test_loss = validate_model(model, val_loader_MLP, nn.MSELoss())
        print(f"Test loss for {model.__class__.__name__} model = {test_loss}")
    else: 
        print(f"Training {model.__class__.__name__} model, {model.__class__.__name__ == 'MLPResidualRegressor'}")
        trained_model = train_model(model, train_loader_MLP, val_loader_MLP, epochs=100, learning_rate=0.0003)
        torch.save(trained_model.state_dict(), f"{model.__class__.__name__}.pt")
        trained_model.eval()
        test_loss = validate_model(model, val_loader_TS, nn.MSELoss())
        print(f"Test loss for {model.__class__.__name__} model = {test_loss}")
    del trained_model
    torch.cuda.empty_cache()
    gc.collect()
    