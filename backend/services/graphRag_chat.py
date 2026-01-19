import re
import pandas as pd

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.context_builder.conversation_history import ConversationHistory

from backend.services.constants import (
    IMG_DEEP_ANALYSIS_SYSTEM_PROMPT,
    CHATBOT_LLM_CONFIG,
    CHATBOT_EMBEDDING_CONFIG,
)


async def chatbot(
    query: str,
    conversation_history: ConversationHistory,
    mode: str,
    user_input_dir: str,
    model_env_value: str,
    embedding_env_value: str,
):
    COMMUNITY_LEVEL = 0

    provider, model_name, _, api_key = model_env_value.split(",")
    chat_config = LanguageModelConfig(
        api_key=api_key,
        type=ModelType.Chat,
        model_provider=provider,
        model=model_name,
        max_retries=20,
    )
    chat_model = ModelManager().get_or_create_chat_model(
        name="local_search",
        model_type=ModelType.Chat,
        config=chat_config,
    )

    provider, model_name, _, api_key = embedding_env_value.split(",")
    embedding_config = LanguageModelConfig(
        api_key=api_key,
        type=ModelType.Embedding,
        model_provider=provider,
        model=model_name,
        max_retries=20,
    )
    text_embedder = ModelManager().get_or_create_embedding_model(
        name="local_search_embedding",
        model_type=ModelType.Embedding,
        config=embedding_config,
    )
    tokenizer = get_tokenizer(chat_config)

    community_df = pd.read_parquet(
        f"{user_input_dir}/communities.parquet", engine="pyarrow"
    )
    community_reports_df = pd.read_parquet(
        f"{user_input_dir}/community_reports.parquet", engine="pyarrow"
    )
    entity_df = pd.read_parquet(f"{user_input_dir}/entities.parquet", engine="pyarrow")
    relationship_df = pd.read_parquet(
        f"{user_input_dir}/relationships.parquet", engine="pyarrow"
    )
    text_unit_df = pd.read_parquet(
        f"{user_input_dir}/text_units.parquet", engine="pyarrow"
    )

    entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
    relationships = read_indexer_relationships(relationship_df)
    reports = read_indexer_reports(community_reports_df, community_df, COMMUNITY_LEVEL)
    text_units = read_indexer_text_units(text_unit_df)

    LANCEDB_URI = f"{user_input_dir}/lancedb"
    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        tokenizer=tokenizer,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 15,
        "conversation_history_user_turns_only": False,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": True,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    model_params = {
        "max_tokens": 5_000,
        "temperature": 0.0,
    }

    if mode == "img_deep_analysis":
        search_engine = LocalSearch(
            model=chat_model,
            context_builder=context_builder,
            system_prompt=IMG_DEEP_ANALYSIS_SYSTEM_PROMPT,
            tokenizer=tokenizer,
            model_params=model_params,
            context_builder_params=local_context_params,
            response_type="bullet points",
        )

    elif mode == "rag":
        search_engine = LocalSearch(
            model=chat_model,
            context_builder=context_builder,
            tokenizer=tokenizer,
            model_params=model_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
        )

    result = await search_engine.search(
        query, conversation_history=conversation_history
    )
    cleaned_response = re.sub(r"\[Data:.*?\]", "", result.response)
    return cleaned_response


# -----------------------------------------------------------------------------
# Development/Testing Entry Point
# -----------------------------------------------------------------------------
# This section is for local development testing only.
# Run with: python -m backend.services.graphRag_chat

if __name__ == "__main__":
    import asyncio

    async def _test_chatbot():
        """Test function for local development."""
        history = [
            {
                "role": "system",
                "content": "TEMPORAL CONTEXT: DAY: Monday, DATE: 1 September 2025, TIME: 13:00. Use these temporal details only if the question requires time awareness—otherwise ignore them.",
            },
        ]

        result = await chatbot(
            user_input_dir="data/tmp/user-1/gRag/output",
            model_env_value=CHATBOT_LLM_CONFIG,
            embedding_env_value=CHATBOT_EMBEDDING_CONFIG,
            query="Why is my alarm set for 16:00 today, did I forget an event?",
            conversation_history=ConversationHistory.from_list(history),
            mode="rag",
        )
        print(result)

    asyncio.run(_test_chatbot())
