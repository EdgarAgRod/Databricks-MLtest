# Databricks notebook source
# MAGIC %md
# MAGIC https://docs.databricks.com/en/getting-started/ml-get-started.html#step-5-preprocess-and-split-the-data

# COMMAND ----------

# MAGIC %md
# MAGIC #Introduction
# MAGIC
# MAGIC ## Get started: Build your first machine learning model on Databricks
# MAGIC This example notebook illustrates how to train a machine learning classification model on Databricks. Databricks Runtime for Machine Learning comes with many libraries pre-installed, including scikit-learn for training and pre-processing algorithms, MLflow to track the model development process, and Hyperopt with SparkTrials to scale hyperparameter tuning.
# MAGIC
# MAGIC In this notebook, you create a classification model to predict whether a wine is considered "high-quality". The dataset[1] consists of 11 features of different wines (for example, alcohol content, acidity, and residual sugar) and a quality ranking between 1 to 10.
# MAGIC
# MAGIC This tutorial covers:
# MAGIC
# MAGIC - Part 1: Train a classification model with MLflow tracking
# MAGIC - Part 2: Hyperparameter tuning to improve model performance
# MAGIC - Part 3: Save results and models to Unity Catalog
# MAGIC For more details on productionizing machine learning on Databricks including model lifecycle management and model inference, see the ML End to End Example (AWS | Azure | GCP).
# MAGIC
# MAGIC [1] The example uses a dataset from the UCI Machine Learning Repository, presented in Modeling wine preferences by data mining from physicochemical properties [Cortez et al., 2009].
# MAGIC
# MAGIC ##Requirements
# MAGIC Cluster running Databricks Runtime 13.3 LTS ML or above

# COMMAND ----------

# MAGIC %md
# MAGIC ##Setup
# MAGIC In this section, you do the following:
# MAGIC
# MAGIC - Configure the MLflow client to use Unity Catalog as the model registry.
# MAGIC - Set the catalog and schema where the model will be registered.
# MAGIC - Read in the data and save it to tables in Unity Catalog.
# MAGIC - Preprocess the data.

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up model registry, catalog, and schema

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# If necessary, replace "main" and "default" with a catalog and schema for which you have the required permissions.
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

white_wine = spark.read.csv("dbfs:/databricks-datasets/wine-quality/winequality-white.csv", sep=';', header=True)
red_wine = spark.read.csv("dbfs:/databricks-datasets/wine-quality/winequality-red.csv", sep=';', header=True)

# Remove the spaces from the column names
# print("Before: ", white_wine.columns)
# print("Before: ", red_wine.columns)
for c in white_wine.columns:
    white_wine = white_wine.withColumnRenamed(c, c.replace(" ", "_"))
for c in red_wine.columns:
    red_wine = red_wine.withColumnRenamed(c, c.replace(" ", "_"))

# Define table names
red_wine_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.red_wine"
white_wine_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.white_wine"

# Write to tables in Unity Catalog
spark.sql(f"DROP TABLE IF EXISTS {red_wine_table}")
spark.sql(f"DROP TABLE IF EXISTS {white_wine_table}")
white_wine.write.saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.white_wine")
red_wine.write.saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.red_wine")


# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-process

# COMMAND ----------

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# Load data from Unity Catalog as Pandas dataframes
white_wine = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.white_wine").toPandas()
red_wine = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.red_wine").toPandas()

# Add Boolean fields for red and white wine
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# Define classification labels based on the wine quality
data_labels = data_df['quality'].astype('int') >= 7
data_df = data_df.drop(['quality'], axis=1)

# Split 80/20 train-test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  data_df,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Training

# COMMAND ----------

# Enable MLflow autologging:
mlflow.autolog()

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
    model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)

    # Models, parameters, and training metrics are tracked automatically
    model.fit(X_train, y_train)

    predicted_probs = model.predict_proba(X_test)
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    roc_curve = sklearn.metrics.RocCurveDisplay.from_estimator(model, X_test, y_test)

    # Save the ROC curve plot to a file
    roc_curve.figure_.savefig("roc_curve.png")

    # The AUC score on test data is not automatically logged, so log it manually
    mlflow.log_metric("test_auc", roc_auc)

    # Log the ROC curve image file as an artifact
    mlflow.log_artifact("roc_curve.png")

    print("Test AUC of: {}".format(roc_auc))


# COMMAND ----------

# MAGIC %md
# MAGIC #Hypeparameter Tuning

# COMMAND ----------

# Define the search space to explore
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def train_model(params):
  # Enable autologging on each worker
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    model_hp = sklearn.ensemble.GradientBoostingClassifier(
      random_state=0,
      **params
    )
    model_hp.fit(X_train, y_train)
    predicted_probs = model_hp.predict_proba(X_test)
    # Tune based on the test AUC
    # In production, you could use a separate validation set instead
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)

    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*roc_auc}

# SparkTrials distributes the tuning using Spark workers
# Greater parallelism speeds processing, but each hyperparameter trial has less information from other trials
# On smaller clusters try setting parallelism=2
spark_trials = SparkTrials(
  parallelism=1
)

with mlflow.start_run(run_name='gb_hyperopt') as run:
  # Use hyperopt to find the parameters yielding the highest AUC
  best_params = fmin(
    fn=train_model,
    space=search_space,
    algo=tpe.suggest,
    max_evals=32,
    trials=spark_trials)


# COMMAND ----------

# Sort runs by their test auc. In case of ties, use the most recent run.
best_run = mlflow.search_runs(
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]
print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))


# COMMAND ----------

model_uri = 'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )

mlflow.register_model(model_uri, f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_quality_model")

