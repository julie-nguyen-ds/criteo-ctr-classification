# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Store

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset

# COMMAND ----------

df_train = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_train_silver")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Features

# COMMAND ----------

# Use _c17 as feature id
mock_id_column_name = "_c17"

# Store categorical feature c19 as is
feature_to_store_c19 = "_c19"

# Store numerical feature c8 + c9 as c89
numerical_feature_8 = "_c8"
numerical_feature_9 = "_c9"
feature_to_store_c89 = "_c89"

# COMMAND ----------

from pyspark.sql.functions import col, first

# Select id and the first occurrence of c2 value for each unique id
df_features_c19 = df_train.groupBy(mock_id_column_name).agg(first(feature_to_store_c19).alias(feature_to_store_c19))
df_features_c89 = df_train.groupBy(mock_id_column_name).agg((first(numerical_feature_8) + first(numerical_feature_9)).alias(feature_to_store_c89))

# COMMAND ----------

# Show the resulting DataFrame
display(df_features_c89.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Store

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists hive_poc_data_science_db

# COMMAND ----------

from databricks.feature_store import feature_table, FeatureLookup, FeatureStoreClient
database_name = "hive_poc_data_science_db"
fs_table_name_ads_c19 = f"{database_name}.ads_features_categorical"
fs_table_name_ads_c89 = f"{database_name}.ads_features_numerical"

fs = FeatureStoreClient()

# # Create a table with features c19
fs.create_table(
    name=fs_table_name_ads_c19, 
    primary_keys=[mock_id_column_name],
    df=df_features_c19,
    description="First occurrences of feature c19 for ID c17",
    tags={"usecase": "classification"}
)

# Create a table with features c89
fs.create_table(
    name=fs_table_name_ads_c89, 
    primary_keys=[mock_id_column_name],
    df=df_features_c89,
    description="Sum of first occurrences of features c8 and c9 for ID c17",
    tags={"usecase": "classification"}
)