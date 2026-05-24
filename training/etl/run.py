import pandas as pd
import os
from ..helpers.GetEnv import GetEnv
from pyspark.sql import SparkSession, functions as F


def create_spark_session():
    return SparkSession.builder.getOrCreate()


def load(df, data_lake_path):
    final_df = df.toPandas()
    final_df.to_csv(f"{data_lake_path}/guardrails_inputs/guardrails_inputs.csv")
    print("Done")


def extract(_env):
    bad_df_raw = pd.read_parquet(
        "hf://datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples/dataset.parquet"
    )
    good_df_raw = pd.read_parquet(
        "hf://datasets/OpenAssistant/oasst2/data"
    )[["text", "lang", "role"]]
    transform(bad_df_raw, good_df_raw, _env)


def transform(bad_df_raw, good_df_raw, _env):
    spark = create_spark_session()

    b_df = spark.createDataFrame(bad_df_raw).withColumns(
        {"label": F.lit(1), "text": F.col("original_sample")}
    ).select("text", "label")

    g_df = spark.createDataFrame(good_df_raw).filter(
        F.col("role") == "prompter"
    ).withColumn("label", F.lit(0)).select("text", "label")

    final_df = b_df.union(g_df).orderBy(F.rand())
    load(final_df, _env["DATA_LAKE_PATH"])


if __name__ == "__main__":
    _env = GetEnv.get_env_variables()
    os.environ["HF_TOKEN"] = _env["HF_TOKEN"]
    extract(_env)
