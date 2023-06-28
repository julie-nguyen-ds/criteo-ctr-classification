# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG mdp_kbank;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- GRANT ALL PRIVILEGES ON CATALOG mdp_kbank TO `julie.nguyen@databricks.com`
# MAGIC -- CREATE DATABASE IF NOT EXISTS poc_data_science_db MANAGED LOCATION 's3://mdp-kbank-poc/data/poc_data_science_db/';

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingest Classification Dataset from S3

# COMMAND ----------

dbutils.fs.ls("s3://mdp-kbank-poc/data-sci-cases/")

# COMMAND ----------

df_test = spark.read.option("delimiter", "\t").csv("s3://mdp-kbank-poc/data-sci-cases/landing/classification-problem/test-dataset")
df_test.write.saveAsTable("mdp_kbank.poc_data_science_db.ads_click_test")

# COMMAND ----------

df_train = spark.read.option("delimiter", "\t").csv("s3://mdp-kbank-poc/data-sci-cases/landing/classification-problem/training-dataset")
df_train.write.saveAsTable("mdp_kbank.poc_data_science_db.ads_click_train")

# COMMAND ----------

df_train = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_train")

# Convert columns from string to integer
def format_ads_click_dataset(df):
    for col_name in df.columns[:14]:
        df = df.withColumn(col_name, df[col_name].cast("int"))
    return df

# Verify the column types
df_train_silver = format_ads_click_dataset(df_train)
df_train_silver.write.saveAsTable("mdp_kbank.poc_data_science_db.ads_click_train_silver")

# COMMAND ----------

df_test = spark.read.table("mdp_kbank.poc_data_science_db.ads_click_test")

# Convert columns from string to integer
def format_ads_click_dataset(df):
    for col_name in df.columns[:14]:
        df = df.withColumn(col_name, df[col_name].cast("int"))
    return df

# Verify the column types
df_test_silver = format_ads_click_dataset(df_test)
df_test_silver.write.saveAsTable("mdp_kbank.poc_data_science_db.ads_click_test_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingest Regression Dataset from S3

# COMMAND ----------

df_regression_test = spark.read.option("header", "true").csv("s3://mdp-kbank-poc/data-sci-cases/landing/regression-problem/test.csv.gz")
df_regression_test.write.saveAsTable("mdp_kbank.poc_data_science_db.nyc_taxi_test")

# COMMAND ----------

df_regression_train = spark.read.option("header", "true").csv("s3://mdp-kbank-poc/data-sci-cases/landing/regression-problem/train.csv.gz")
df_regression_train.write.saveAsTable("mdp_kbank.poc_data_science_db.nyc_taxi_train")

# COMMAND ----------

display(spark.read.table("poc_data_science_db.nyc_taxi_train"))

# COMMAND ----------


