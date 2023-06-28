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
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import databricks.automl as db_automl
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup, FeatureStoreClient

import mlflow
from mlflow.models import infer_signature

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 0. Prepare Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC #### a. Load Dataset

# COMMAND ----------

df_train = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_train_silver")
df_test = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_test_silver")


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

df_train = process_dataset(df_train).dropna()
df_test = process_dataset(df_test).dropna()

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

# fs.create_training_set will look up features in model_feature_lookups with matched key
training_set = fs.create_training_set(
    df_train, 
    feature_lookups=feature_lookups,
    label=target
)

# fs.create_training_set will look up features in model_feature_lookups with matched key
test_set = fs.create_training_set(
    df_test, 
    feature_lookups=feature_lookups,
    label=target
)

df_fs_test = test_set.load_df().dropna()
df_fs_train = training_set.load_df().dropna()

# COMMAND ----------

df_fs_test.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Model with AutoML

# COMMAND ----------

df_subset = df_fs_train.dropna()

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

import numpy as np
import math
from pyspark.ml.classification import RandomForestClassifier
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
# MAGIC #### b. Performance Training (150M row)

# COMMAND ----------

# Train subset with Stratified Split
df_subset = df_fs_train.dropna()
df_clicked = df_subset.select("*").filter(df_subset._c0 == 1).limit(14000000)
df_non_clicked = df_subset.select("*").filter(df_subset._c0 == 0).limit(140000000)
df_subset = df_clicked.union(df_non_clicked)

# Test subset
df_test_subset = df_fs_test.dropna().limit(140000)

display(df_subset)

# COMMAND ----------

# Past Run
df_subset_repartitioned = df_subset.repartition(numPartitions=384)
display(df_subset_repartitioned)

# COMMAND ----------

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
    
    predictions = model.transform(df_test_subset)

    # Create both evaluators
    evaluator_multi = MulticlassClassificationEvaluator(labelCol="_c0")
    evaluator = BinaryClassificationEvaluator().setLabelCol("_c0")

    # Get metrics
    accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
    weighted_precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
    weighted_recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    mlflow.log_metrics({"test_accuracy": accuracy,
                        "test_f1": f1,
                        "test_auc": auc,
                        "test_precision": weighted_precision,
                        "test_recall": weighted_recall
                        })

# COMMAND ----------

# MAGIC %md 
# MAGIC #### c. Balanced Class Training (28M row)

# COMMAND ----------

# Train subset with Stratified Split
df_subset = df_fs_train.dropna()
df_clicked = df_subset.select("*").filter(df_subset._c0 == 1).limit(14000000)
df_non_clicked = df_subset.select("*").filter(df_subset._c0 == 0).limit(14000000)
df_subset = df_clicked.union(df_non_clicked)

# Test subset
df_test_subset = df_fs_test.dropna().limit(140000)

# COMMAND ----------

# Past Run
df_subset_repartitioned = df_subset.repartition(numPartitions=384)

# COMMAND ----------

# 140M + 14M + repartition(384) = (12 workers)
# 140M + 14M + repartition(192) = 1.2hours (6 workers)
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
    
    predictions = model.transform(df_test_subset)

    # Create both evaluators
    evaluator_multi = MulticlassClassificationEvaluator(labelCol="_c0")
    evaluator = BinaryClassificationEvaluator().setLabelCol("_c0")

    # Get metrics
    accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
    weighted_precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
    weighted_recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    mlflow.log_metrics({"test_accuracy": accuracy,
                        "test_f1": f1,
                        "test_auc": auc,
                        "test_precision": weighted_precision,
                        "test_recall": weighted_recall
                        })

# COMMAND ----------

