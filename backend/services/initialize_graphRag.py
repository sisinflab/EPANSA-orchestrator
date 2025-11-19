import subprocess
from neo4j import GraphDatabase
import pandas as pd
import os

from backend.services.constants import (COMMUNITIES_QUERY,
                                        ENTITIES_QUERY,
                                        TEXT_UNIT_QUERY,
                                        RELATIONSHIPS_QUERY)


def run_df(driver, query, params=None):
    with driver.session() as sess:
        records = sess.run(query, params or {})
        rows = [r.data() for r in records]
    return pd.DataFrame(rows)


def create_parquet(
        uri: str = "bolt://localhost:7687",
        userName: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        user_dir: str = None,
):
    driver = GraphDatabase.driver(uri=uri, auth=(userName, password), database=database)

    gRag_out_dir = f"{user_dir}/gRag/output"
    if not os.path.isdir(gRag_out_dir):
        os.makedirs(gRag_out_dir)

    try:
        entities_df = run_df(driver, ENTITIES_QUERY)
        entities_df.to_parquet(f"{gRag_out_dir}/entities.parquet", index=False, engine="pyarrow")

        relationships_df = run_df(driver, RELATIONSHIPS_QUERY)
        relationships_df.to_parquet(f"{gRag_out_dir}/relationships.parquet", index=False, engine="pyarrow")

        text_units_df = run_df(driver, TEXT_UNIT_QUERY)
        text_units_df.to_parquet(f"{gRag_out_dir}/text_units.parquet", index=False, engine="pyarrow")

        communities_df = run_df(driver, COMMUNITIES_QUERY)
        communities_df.to_parquet(f"{gRag_out_dir}/communities.parquet", index=False, engine="pyarrow")

    except Exception as e:
        raise e


def perform_indexing(user_dir: str, model_env_value: str, embedding_env_value: str):
    provider, model_name, _, api_key = model_env_value.split(",")
    provider_e, model_name_e, _, api_key_e = embedding_env_value.split(",")

    gRag_dir = f"{user_dir}/gRag"
    with open(f"{gRag_dir}/settings.yaml", "w") as f:
        settings_yaml = f"""
        workflows:
          - create_community_reports
          - generate_text_embeddings

        paths:
          input: ./input
          output: ./output
        
        models:
          default_chat_model:
            model_provider: {provider}
            api_key: {api_key}
            type: chat
            model: {model_name}
            max_retries: 2
        
          default_embedding_model:
            model_provider: {provider_e}
            api_key: {api_key_e}
            type: embedding
            model: {model_name_e}
            max_retries: 2
        
        community_reports:
          level: 2
          use_summaries: true
          map_max_tokens: 1000
          reduce_max_tokens: 2000
          temperature: 0.0
        """
        f.write(settings_yaml)

    try:
        subprocess.run(
            ["graphrag", "index", "--root", str(gRag_dir)],
            capture_output=True,
            text=True
        )
    except Exception as e:
        raise e
