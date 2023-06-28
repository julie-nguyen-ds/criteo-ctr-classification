# Databricks notebook source
# MAGIC %md
# MAGIC # Criteo Ads Click Classification 
# MAGIC This notebook shows how to train a model for Criteo Ads Click classification with MLFlow, Databricks AutoML and various methods. 
# MAGIC
# MAGIC This showcases: 
# MAGIC - 1. AutoML
# MAGIC - 1. Random Forest Model + Hyperparameter Tuning with Grid Search / Random Search
# MAGIC - 2. Logging MLFlow experiments
# MAGIC - 3. Model Versioning in MLFlow Registry
# MAGIC - 4. Batch Inference
# MAGIC - 8. Feature Store Usage

# COMMAND ----------

import os
import time

from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import date_format, datediff, to_date, udf, count, countDistinct
from pyspark.ml.feature import FeatureHasher, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 0. Prepare Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC #### a. Load Dataset

# COMMAND ----------

df_train = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_train_silver")

# COMMAND ----------

df_train.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### b. Process Dataset

# COMMAND ----------

numerical_features = ['_c1', '_c2', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c89']
categorical_features = ['_c12', '_c13', '_c14', '_c15', '_c16', '_c17', '_c18', '_c19', '_c20', '_c21', '_c22', '_c23', '_c24', '_c25', '_c26', '_c27', '_c28', '_c29', '_c30', '_c31', '_c32', '_c33', '_c34', '_c35', '_c36', '_c37', '_c38', '_c39']

partly_empty_numerical_features = ['_c4', '_c5', '_c11']
partly_empty_categorical_features = ['_c27', '_c29', '_c30', '_c36']
categorical_features_high_cardinality = ['_c12', '_c14', '_c15', '_c16', '_c18','_c23', '_c24', '_c25', '_c33', '_c34', '_c35', '_c36']
features_to_drop = ['_c19']

training_numerical_features = [col for col in numerical_features if col not in partly_empty_numerical_features]
training_categorical_features = [col for col in categorical_features if col not in categorical_features_high_cardinality + features_to_drop + partly_empty_categorical_features]

# COMMAND ----------

def process_dataset(df, 
                    numerical_features=numerical_features, 
                    categorical_features=categorical_features, 
                    partly_empty_numerical_features=partly_empty_numerical_features, partly_empty_categorical_features=partly_empty_categorical_features,
                    categorical_features_high_cardinality=categorical_features_high_cardinality,
                    features_to_drop=features_to_drop):
    
    # Drop features if they have more than 30% null values 
    partly_empty_features = partly_empty_numerical_features + partly_empty_categorical_features
    df_train = df.drop(*partly_empty_features) 

    # Drop features having a too high cardinality (>10k categories)
    df_train = df_train.drop(*categorical_features_high_cardinality) 

    # Drop features that we will fetch in the feature store
    df_train = df_train.drop(*features_to_drop) 
    
    return df_train

# COMMAND ----------

df_train = process_dataset(df_train)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 8. Fetch Feature Store Table

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup, FeatureStoreClient

target = "_c0"
mock_id = "_c17"
database_name = "hive_poc_data_science_db"
fs_table_name_ads_c19 = f"{database_name}.ads_features_categorical"
fs_table_name_ads_c89 = f"{database_name}.ads_features_numerical"

fs = FeatureStoreClient()

feature_lookups = [
      FeatureLookup(
          table_name=fs_table_name_ads_c19,
          lookup_key=mock_id
      ),
      FeatureLookup(
          table_name=fs_table_name_ads_c89,
          lookup_key=mock_id,
      )
]

# fs.create_training_set will look up features in model_feature_lookups with matched key from training_labels_df
training_set = fs.create_training_set(
    df_train, 
    feature_lookups=feature_lookups,
    label=target
)

training_pd = training_set.load_df()
display(training_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Model with AutoML

# COMMAND ----------

df_subset = training_pd.dropna()

# COMMAND ----------

# Stratified split
df_clicked = df_subset.select("*").filter(df_subset._c0 == 1).limit(50000)
df_non_clicked = df_subset.select("*").filter(df_subset._c0 == 0).limit(50000)
df_subset = df_clicked.union(df_non_clicked)


# COMMAND ----------

df_subset.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Train Model & Log Experiments

# COMMAND ----------

import databricks.automl as db_automl

summary_cl = db_automl.classify(df_subset, target_col=target, primary_metric="log_loss", timeout_minutes=15, experiment_dir = "/Users/julie.nguyen@databricks.com/databricks_automl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Register Model

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

# Create sample input to be logged
df_sample = df_subset.limit(10).toPandas()
x_sample = df_sample.drop(columns=[target])
y_sample = df_sample[target]

# Get the model created by AutoML 
best_model = summary_cl.best_trial.load_model()
model_name = "automl_click_model"

env = mlflow.pyfunc.get_default_conda_env()
with open(mlflow.artifacts.download_artifacts("runs:/" + summary_cl.best_trial.mlflow_run_id + "/model/requirements.txt"), 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

# Create a new run in the same experiment as our automl run.
with mlflow.start_run(run_name="best_automl_classification_model", experiment_id=summary_cl.experiment.experiment_id) as run:
  
  # Use the feature store client to log our best model
  fs.log_model(
              model=best_model,
              artifact_path="model", 
              flavor=mlflow.sklearn, # flavour of the model (our LightGBM model has a SkLearn Flavour)
              training_set=training_set, # training set you used to train your model with AutoML
              input_example=x_sample, # example of the dataset, should be Pandas
              signature=infer_signature(x_sample, y_sample), # schema of the dataset, not necessary with FS, but nice to have 
              conda_env = env
          )
  mlflow.log_metrics(summary_cl.best_trial.metrics)
  mlflow.log_params(summary_cl.best_trial.params)
  
model_registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

# Move the model in production
client = mlflow.tracking.MlflowClient()
print("Registering model version " + model_registered.version + " as production model")
client.transition_model_version_stage(model_name, model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Run Batch Inference

# COMMAND ----------

df_test = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_test_silver")
df_test = process_dataset(df_test).dropna()

# COMMAND ----------

## For sake of simplicity
model_name = "automl_click_model"
scored_df = fs.score_batch(f"models:/{model_name}/Production", df_test, result_type="string")
display(scored_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Train Model Manually

# COMMAND ----------

# MAGIC %md 
# MAGIC #### a. Grid Search

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
import numpy as np
import math
from pyspark.ml.tuning import CrossValidatorModel, CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Convert string features to numerical using StringIndexer
string_indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep").fit(df_subset) for col in training_categorical_features
]

# Assemble the features into a vector column
assembler = VectorAssembler(
    inputCols=training_numerical_features + [f"{col}_index" for col in training_categorical_features],
    outputCol="features"
)

# Create a Random Forest classifier
rf = RandomForestClassifier(labelCol="_c0", maxBins=10000, featuresCol="features")

# Create a pipeline to chain the stages together
pipeline = Pipeline(stages=string_indexers + [assembler, rf])

paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [5, 3]).build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator().setLabelCol("_c0"),
                          numFolds=2) 

cvModel = crossval.fit(df_subset)

# COMMAND ----------

display(df_subset)

# COMMAND ----------

# MAGIC %md
# MAGIC #### b. Performance Experiment

# COMMAND ----------

df_subset = training_pd.dropna()
print(f"Initial dataset had {training_pd.count()} rows while dropping na makes it down to {df_subset.count()} rows.")

# COMMAND ----------

training_pd.rdd.getNumPartitions()
df_subset.rdd.getNumPartitions()

# COMMAND ----------

# Stratified split
#df_subset = training_pd.dropna()
df_clicked = df_subset.select("*").filter(df_subset._c0 == 1).limit(14000000)
df_non_clicked = df_subset.select("*").filter(df_subset._c0 == 0).limit(14000000)
df_subset = df_clicked.union(df_non_clicked)
display(df_subset)

# COMMAND ----------

df_subset.count()

# COMMAND ----------

df_subset.rdd.getNumPartitions()

# COMMAND ----------

# Past Run
df_subset_repartitioned = df_subset.repartition(numPartitions=128)
print(f"Repartitioned Dataframe has {df_subset_repartitioned.rdd.getNumPartitions()} partitions.")
display(df_subset_repartitioned)

# COMMAND ----------

# 140M + 14M + repartition(128) = 1.29hours
# 140M + 14M + repartition(192) = 1.1hours
# 26M + repartition(120) = 13 min
# 10M rows + repartition(60) = 5.6min

from mlflow.models import infer_signature
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import mlflow

mlflow.spark.autolog(silent=True)

# Create sample input to be logged
df_sample = df_subset_repartitioned.limit(10).toPandas()
x_sample = df_sample.drop(columns=[target])
y_sample = df_sample[target]

# Get the model created by AutoML 
model_name = "automl_click_model"
env = mlflow.pyfunc.get_default_conda_env()

with mlflow.start_run():
    # Convert string features to numerical using StringIndexer
    string_indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep").fit(df_subset_repartitioned) for col in training_categorical_features
    ]

    # Assemble the features into a vector column
    assembler = VectorAssembler(
        inputCols=training_numerical_features + [f"{col}_index" for col in training_categorical_features],
        outputCol="features"
    )

    # Create a Random Forest classifier
    rf = RandomForestClassifier(labelCol="_c0", maxBins=30000, featuresCol="features")

    # Create a pipeline to chain the stages together
    pipeline = Pipeline(stages=string_indexers + [assembler, rf])

    # Fit the pipeline to the DataFrame
    model = pipeline.fit(df_subset_repartitioned)
    fs.log_model(model, 
                 artifact_path="model", 
                 flavor=mlflow.spark,
                 training_set=training_set,
                 input_example=x_sample,
                 signature=infer_signature(x_sample, y_sample),
                 conda_env = env
                 )
    
    mlflow.log_tag("gpu", "True")
    # mlflow.log_metrics(summary_cl.best_trial.metrics)
    # mlflow.log_params(summary_cl.best_trial.params)

# COMMAND ----------

display(df_subset)

# COMMAND ----------

# 26M row: 9min to index and vectorize + 14min training
# 10M row: 5min to index and vectorize + 2min training
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import mlflow

mlflow.spark.autolog(silent=True)

# Create sample input to be logged
df_sample = df_subset.limit(10).toPandas()
x_sample = df_sample.drop(columns=[target])
y_sample = df_sample[target]

# Get the model created by AutoML 
model_name = "automl_click_model"
env = mlflow.pyfunc.get_default_conda_env()
with open(mlflow.artifacts.download_artifacts("runs:/" + summary_cl.best_trial.mlflow_run_id + "/model/requirements.txt"), 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')


with mlflow.start_run():
    # Convert string features to numerical using StringIndexer
    string_indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep").fit(df_subset) for col in training_categorical_features
    ]

    # Assemble the features into a vector column
    assembler = VectorAssembler(
        inputCols=training_numerical_features + [f"{col}_index" for col in training_categorical_features],
        outputCol="features"
    )

    # Create a Random Forest classifier
    rf = RandomForestClassifier(labelCol="_c0", maxBins=12000, featuresCol="features")

    # Create a pipeline to chain the stages together
    pipeline = Pipeline(stages=string_indexers + [assembler, rf])

    # Fit the pipeline to the DataFrame
    model = pipeline.fit(df_subset)
    fs.log_model(model, 
                 artifact_path="model", 
                 flavor=mlflow.spark,
                 training_set=training_set,
                 input_example=x_sample,
                 signature=infer_signature(x_sample, y_sample),
                 conda_env = env
                 )

    # mlflow.log_metrics(summary_cl.best_trial.metrics)
    # mlflow.log_params(summary_cl.best_trial.params)

# COMMAND ----------

# 100k rows

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import mlflow

mlflow.spark.autolog(silent=True)

# Create sample input to be logged
df_sample = df_subset.limit(10).toPandas()
x_sample = df_sample.drop(columns=[target])
y_sample = df_sample[target]

# Get the model created by AutoML 
model_name = "automl_click_model"
env = mlflow.pyfunc.get_default_conda_env()
with open(mlflow.artifacts.download_artifacts("runs:/" + summary_cl.best_trial.mlflow_run_id + "/model/requirements.txt"), 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')


with mlflow.start_run():
    # Convert string features to numerical using StringIndexer
    string_indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep").fit(df_subset) for col in training_categorical_features
    ]

    # Assemble the features into a vector column
    assembler = VectorAssembler(
        inputCols=training_numerical_features + [f"{col}_index" for col in training_categorical_features],
        outputCol="features"
    )

    # Create a Random Forest classifier
    rf = RandomForestClassifier(labelCol="_c0", maxBins=10000, featuresCol="features")

    # Create a pipeline to chain the stages together
    pipeline = Pipeline(stages=string_indexers + [assembler, rf])

    # Fit the pipeline to the DataFrame
    model = pipeline.fit(df_subset)
    fs.log_model(model, 
                 artifact_path="model", 
                 flavor=mlflow.spark,
                 training_set=training_set,
                 input_example=x_sample,
                 signature=infer_signature(x_sample, y_sample),
                 conda_env = env
                 )

    # mlflow.log_metrics(summary_cl.best_trial.metrics)
    # mlflow.log_params(summary_cl.best_trial.params)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC #### b. Performance Training (700M row)