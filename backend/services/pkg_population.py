import logging
import os
from typing import Optional
from libs.llm_graph_builder.functions import (
    upload_file,
    delete_document_and_entities,
    extract_knowledge_graph_from_file,
    post_processing,
    compute_entity_embeddings,
    execute_cypher_query,
)
from backend.services.img_description import get_img_description
from backend.services.constants import (
    QUERY_TO_MERGE_KGs,
    QUERY_TO_CONNECT_TO_USER,
    QUERY_TO_DELETE_USER_CONNECTION,
    TMP_DIR,
    URI,
    USERNAME,
    PASSWORD,
    EXTRACTION_MODEL_NAME,
    EXTRACTION_LLM_CONFIG,
)
import backend.services.initialize_graphRag as initialize_graphRag


# ----Utils (to call llm-graph-builder functionalities)-----------------------------------------------------------------------
async def llm_kg_builder_upload(fileName, database):
    return await upload_file(
        uri=URI,
        userName=USERNAME,
        password=PASSWORD,
        database=database,
        fileName=fileName,
        model_name=EXTRACTION_MODEL_NAME,
    )


async def llm_kg_builder_delete(fileName, database):
    try:
        await llm_kg_builder_cypher(
            QUERY_TO_DELETE_USER_CONNECTION,
            params={"fileName": os.path.splitext(fileName)[0]},
            database=database,
        )
    except Exception as e:
        raise Exception(
            f"Error while deleting User connections for {fileName}: {str(e)}"
        )

    return await delete_document_and_entities(
        fileName=fileName,
        uri=URI,
        userName=USERNAME,
        password=PASSWORD,
        database=database,
    )


async def llm_kg_builder_extract(
    fileName,
    database,
    content,
    additional_instructions=None,
    embedding_model=None,
    embedding_dimension=None,
):
    return await extract_knowledge_graph_from_file(
        file_name=fileName,
        content=content,
        additional_instructions=additional_instructions,
        uri=URI,
        userName=USERNAME,
        password=PASSWORD,
        database=database,
        model_name=EXTRACTION_MODEL_NAME,
        model_env_value=EXTRACTION_LLM_CONFIG,
        token_chunk_size=int(os.getenv("TOKENS_PER_CHUNK", 300)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 20)),
        chunks_to_combine=int(os.getenv("NUMBER_OF_CHUNKS_TO_COMBINE", 3)),
        max_token_chunk_size=int(os.getenv("MAX_TOKEN_CHUNK_SIZE", 10000)),
        words_for_big_file=int(os.getenv("WORDS_FOR_BIG_FILE", 2000)),
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
    )


async def llm_kg_builder_postprocess(database, embedding_model, embedding_dimension):
    uri = URI
    userName = USERNAME
    password = PASSWORD
    database = database
    embedding_model = embedding_model
    embedding_dimension = embedding_dimension
    user_dir = f"{TMP_DIR}/{database}"

    res = await post_processing(
        uri, userName, password, database, embedding_model, embedding_dimension
    )

    logging.info("Creating .parquet files for GraphRAG...")
    initialize_graphRag.create_parquet(
        uri, userName, password, database, user_dir=user_dir
    )

    logging.info("Performing indexing for GraphRAG...")
    initialize_graphRag.perform_indexing(
        user_dir=user_dir,
        model_env_value=os.getenv("COMMUNITIES_LLM_CONFIG"),
        embedding_env_value=os.getenv("CHATBOT_EMBEDDING_CONFIG"),
    )
    return res


async def llm_kg_builder_compute_embeddings(database, embedding_model):
    return await compute_entity_embeddings(
        uri=URI,
        userName=USERNAME,
        password=PASSWORD,
        database=database,
        embedding_model=embedding_model,
    )


async def llm_kg_builder_cypher(query, params, database):
    return await execute_cypher_query(
        uri=URI,
        userName=USERNAME,
        password=PASSWORD,
        database=database,
        query=query,
        params=params,
    )


async def insertion_pipeline(
    fileName,
    database,
    content,
    additional_instructions=None,
    embedding_model=None,
    embedding_dimension=None,
):
    """upload -> extract KG -> post-process (or delete)"""

    res = await llm_kg_builder_upload(fileName, database)
    if res["status"] == "Success":
        logging.info(f"Extracting KG for file {fileName}...")
        res = await llm_kg_builder_extract(
            fileName,
            database,
            content,
            additional_instructions,
            embedding_model,
            embedding_dimension,
        )
        if res["status"] == "Success":
            logging.info(f"Proceding with post-processing for file {fileName}...")
            await llm_kg_builder_postprocess(
                database, embedding_model, embedding_dimension
            )
            return res
        else:
            logging.info(
                f"deleting created entities for {fileName} due to extraction failure..."
            )
            await llm_kg_builder_delete(fileName, database)
            return res
    else:
        logging.info(
            f"deleting created entities for {fileName} due to uploading failure..."
        )
        await llm_kg_builder_delete(fileName, database)
        return res


async def update_pipeline(
    fileName,
    database,
    content,
    additional_instructions=None,
    embedding_model=None,
    embedding_dimension=None,
):
    """delete prev entities -> upload -> extract KG -> post-process (or delete)"""

    logging.info(f"deleting entities for {fileName} (to be updated)...")
    res = await llm_kg_builder_delete(fileName, database)

    if res["status"] == "Success":
        res = await llm_kg_builder_upload(fileName, database)
        if res["status"] == "Success":
            logging.info(f"Extracting KG for file {fileName}...")
            res = await llm_kg_builder_extract(
                fileName,
                database,
                content,
                additional_instructions,
                embedding_model,
                embedding_dimension,
            )
            if res["status"] == "Success":
                logging.info(f"Proceeding with post-processing for file {fileName}...")
                await llm_kg_builder_postprocess(
                    database, embedding_model, embedding_dimension
                )
                return res
            else:
                logging.info(
                    f"deleting created entities for {fileName} due to extraction failure..."
                )
                await llm_kg_builder_delete(fileName, database)
                return res
        else:
            logging.info(
                f"deleting created entities for {fileName} due to uploading failure..."
            )
            await llm_kg_builder_delete(fileName, database)
            return res
    else:
        logging.info(
            f"ATTENTION! Entities for file {fileName} have been deleted and not re-created!"
        )
        return res


# ----Main------------------------------------------------------------------------
async def extract_kg_triples(
    json_path: str = None,
    unstruct_path: Optional[str] = None,
    operation: str = None,
    database: str = None,
    kind: str = None,
    embedding_model=None,
    embedding_dimension: int = None,
):
    """Extracts knowledge graph triples from file (events, alarms, etc.) and post-processes them."""

    with open(json_path, "rb") as f:
        json_content = f.read()
    fileName = os.path.basename(json_path)

    # The kind of operation (INSERT or UPDATE) will affect only the first part of the pipeline; following steps are the same
    if operation == "insert":
        logging.info(f"Starting metadata INSERTION for {fileName}...")
        res = await insertion_pipeline(
            fileName,
            database,
            json_content,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )
    elif operation == "update":
        logging.info(f"Starting metadata UPDATE for {fileName}...")
        res = await update_pipeline(
            fileName,
            database,
            json_content,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )
    else:
        raise ValueError("Operation must be 'insert' or 'update'")

    # Connect the extracted KG to the User node
    try:
        await llm_kg_builder_cypher(
            QUERY_TO_CONNECT_TO_USER,
            params={
                "fileName": os.path.splitext(fileName)[0],
                "relation": f"HAS_{kind.upper()}",
            },
            database=database,
        )
    except Exception as e:
        raise Exception(f"Error while connecting {fileName} KG to User node: {str(e)}")

    # STRUCTURED kind of contents (metadata-only)
    if res["status"] == "Success" and not unstruct_path:
        logging.info(f"PKG updated for {fileName}.")
        os.remove(json_path)

    # UNSTRUCTURED kind of contents
    elif res["status"] == "Success":
        fileName_u = os.path.basename(unstruct_path)
        ext_u = os.path.splitext(fileName_u)[1].lower()

        # NOTES
        if ext_u == ".txt":
            with open(unstruct_path, "rb") as f:
                unstruct_content = f.read()

            logging.info(
                f"Starting KG extraction for the content of the note {fileName}..."
            )
            r = await insertion_pipeline(
                fileName_u,
                database,
                unstruct_content,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
            if r["status"] == "Success":
                os.remove(unstruct_path)
            else:
                raise Exception(
                    f"Error while extracting KG from unstructured content of the note {fileName}"
                )

        # PHOTOS
        elif ext_u == ".png" or ext_u == ".jpg" or ext_u == ".jpeg":
            logging.info(f"Calling Chain-of-Experts for {fileName_u}...")

            VL_model_env_value = os.getenv("IMG_ANALYSIS_VL_CONFIG")
            img_description = get_img_description(
                question="Describe this image in detail",
                img_path=unstruct_path,
                VL_model_env_value=VL_model_env_value,
            )

            new_unstruct_path = os.path.splitext(unstruct_path)[0] + ".txt"
            with open(new_unstruct_path, "wb") as image_file:
                img_description = img_description.encode("utf-8")
                image_file.write(img_description)

            with open(new_unstruct_path, "rb") as f:
                img_description = f.read()

            logging.info(
                f"Starting KG extraction for the content of the image {fileName_u}..."
            )

            r = await insertion_pipeline(
                fileName_u,
                database,
                img_description,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
            if r["status"] == "Success":
                os.remove(new_unstruct_path)
            else:
                os.remove(new_unstruct_path)
                raise Exception(
                    f"Error while extracting KG from unstructured content of the image {fileName_u}"
                )

        # DOCs
        elif ext_u == ".pdf" or ext_u == ".docx":
            with open(unstruct_path, "rb") as f:
                unstruct_content = f.read()

            logging.info(
                f"Starting KG extraction for the content of the document {fileName_u}..."
            )
            r = await insertion_pipeline(
                fileName_u,
                database,
                unstruct_content,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
            if r["status"] == "Success":
                os.remove(unstruct_path)
            else:
                raise Exception(
                    f"Error while extracting KG from unstructured content of the document {fileName_u}"
                )

        # Unsupported file formats
        else:
            raise ValueError(f"Unsupported file format for {fileName_u}")

        try:
            # Union of the two sub-KGs (metadata + content)
            await llm_kg_builder_cypher(
                QUERY_TO_MERGE_KGs,
                params={"fileName": fileName, "new_fileName_u": fileName_u},
                database=database,
            )
            os.remove(json_path)
            logging.info(
                f"Both unstructured content and metadata added to PKG and connected to each-other for {fileName}."
            )
        except Exception as e:
            raise Exception(
                f"Error while merging structured and unstructured sub-KGs for {fileName}: {str(e)}"
            )

    else:
        logging.info(f"PKG update for {fileName} FAILED.")
        return {"status": "Failed", "message": "Operation failed: PKG not updated"}

    message = f"Operation completed successfully: PKG updated for {fileName}"
    logging.info(message)
    return {"status": "Success", "message": message}


async def delete_kg_triples(
    fileName: str = None,
    database: str = None,
    embedding_model=None,
    embedding_dimension=None,
):
    """Deletes all the triples related to a file (events, alarms, etc.) from the KG."""
    res = await llm_kg_builder_delete(fileName, database)
    await llm_kg_builder_postprocess(database, embedding_model, embedding_dimension)
    return {"status": res["status"], "message": res["message"]}


# -----------------------------------------------------------------------------
# Development/Testing Entry Point
# -----------------------------------------------------------------------------
# This section is for local development testing only.
# Run with: python -m backend.services.pkg_population

if __name__ == "__main__":
    import asyncio

    async def _test_extraction():
        """Test function for local development."""
        from libs.llm_graph_builder.src.shared.common_fn import load_embedding_model

        embedding_model, embedding_dimension = load_embedding_model()

        await extract_kg_triples(
            json_path="data/tmp/user-1/note_13.txt",
            unstruct_path="data/tmp/user-1/noteContent_13.txt",
            operation="update",
            database="user-1",
            kind="note",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

    asyncio.run(_test_extraction())
