import os
import logging
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from neo4j import GraphDatabase
from libs.llm_graph_builder.src.llm import get_llm
from pydantic import BaseModel, Field

USER_PROFILE_AND_LOCATION_QUERY = """
// --- phase 1: get last known location from latest photo ---
MATCH (u:User {userId: $userId})
OPTIONAL MATCH (u)-[:HAS_PHOTO]->(p:Photo)
// order
WITH u, p ORDER BY p.creation_date DESC, p.creation_time DESC
OPTIONAL MATCH (p)-[:HAS_LOCATION]->(loc:Location)
WITH u, head(collect(loc.id)) as last_known_location_str

// --- phase 2: get user facts and visited locations ---
MATCH (u:User {userId: $userId})
OPTIONAL MATCH (u)-[:HAS_PHOTO]->()-[:HAS_LOCATION]->(visited_loc:Location)
OPTIONAL MATCH (u)-[:HAS_EVENT]->(e:Event)-[:HAS_LABEL]->(e_label:Label)
OPTIONAL MATCH (u)-[:HAS_NOTE]->(n:Note)-[:HAS_LABEL]->(n_label:Label)

WITH u, last_known_location_str,
     // facts about visited locations and interests
     collect(DISTINCT "User has visited '" + visited_loc.id + "'.") +
     collect(DISTINCT "User is interested in '" + e_label.id + "'.") +
     collect(DISTINCT "User is interested in '" + n_label.id + "'.") AS facts,
     // get unique visited locations
     collect(DISTINCT visited_loc) AS locations

RETURN 
    last_known_location_str, 
    apoc.text.join(facts, " ") as user_profile_text,
    // extract city names from location IDs
    [loc IN locations WHERE loc IS NOT NULL | 
        trim(
            CASE 
                WHEN size(split(loc.id, ',')) >= 3 THEN split(loc.id, ',')[size(split(loc.id, ','))-2]
                WHEN size(split(loc.id, ',')) = 2 THEN split(loc.id, ',')[1]
                ELSE split(loc.id, ',')[0] 
            END
        )
    ] as visited_cities_list
"""

USER_PROFILE_SUMMARIZATION_PROMPT_TEMPLATE = """
You are a nuanced user profiling expert for a recommender system.
Based on the following facts about a user, create two distinct, rich, and **well-balanced** narrative summaries.
Identify all the user's main interests (e.g., culture, sports, nature, food) and give them **proportional weight**. Do not let a single interest dominate the entire profile if other interests are present.
Each summary MUST be a single paragraph and MUST NOT exceed 150 words.

1.  A summary focused ONLY on the user's preferences for FOOD, dining, restaurants, and nightlife (max 150 words).
2.  A summary focused ONLY on the user's preferences for TRAVEL, including places to visit, sightseeing, culture, nature, and activities (max 150 words).

Return a JSON object with the keys "food_summary" and "travel_summary".

Facts: {user_profile_text}
"""


# Pydantic model for structured output
class UserProfileSummaries(BaseModel):
    food_summary: str = Field(
        description="A narrative summary of the user's preferences for FOOD, dining, and nightlife.")
    travel_summary: str = Field(
        description="A narrative summary of the user's preferences for TRAVEL, sightseeing, culture, nature, and activities.")


async def get_or_create_user_embedding(database: str, user_id: int | str, embedding_model, model_env_value: str):
    driver = None
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

        with driver.session(database=database) as session:
            logging.info(f"Extracting or creating profile for user {user_id} from knowledge graph...")
            result = session.run(USER_PROFILE_AND_LOCATION_QUERY, userId=int(user_id)).single()

            # MERGE ensures a result is always returned, but as a safeguard.
            if not result:
                logging.error(f"Failed to get a result for user {user_id} even with MERGE. This should not happen.")
                # Fallback to creating a completely empty profile.
                profile_text = ""
                last_known_location = None
                visited_cities = []
            else:
                profile_text = result.get('user_profile_text')
                last_known_location = result.get('last_known_location_str')
                visited_cities = list(set(result.get('visited_cities_list', [])))

            # Handle new or empty users gracefully.
            if not profile_text or not str(profile_text).strip():
                logging.warning(f"No profile text found for user {user_id}. Creating default empty profile.")
                # For new or empty users, create empty summaries and generate neutral embeddings.
                food_summary = ""
                travel_summary = ""
                # Embed empty strings to get a neutral, non-None vector.
                food_embedding = embedding_model.embed_query(food_summary)
                travel_embedding = embedding_model.embed_query(travel_summary)
            else:
                # generate one summary for food and one for travel
                llm, _ = get_llm(model_env_value)
                parser = JsonOutputParser(pydantic_object=UserProfileSummaries)
                prompt = ChatPromptTemplate.from_template(
                    template=USER_PROFILE_SUMMARIZATION_PROMPT_TEMPLATE,
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )
                chain = prompt | llm | parser

                response_json = await chain.ainvoke({"user_profile_text": profile_text})

                food_summary = response_json.get('food_summary', '')
                travel_summary = response_json.get('travel_summary', '')

                logging.info("=" * 80)
                logging.info(f"Generated Food Profile Summary for User ID {user_id}:\n{food_summary}")
                logging.info("-" * 80)
                logging.info(f"Generated Travel Profile Summary for User ID {user_id}:\n{travel_summary}")
                logging.info("=" * 80)

                # Create two separate embeddings
                food_embedding = embedding_model.embed_query(food_summary) if food_summary else embedding_model.embed_query("")
                travel_embedding = embedding_model.embed_query(travel_summary) if travel_summary else embedding_model.embed_query("")

            # Store both embeddings in Neo4j, regardless of whether they are new or updated.
            store_query = """
            MATCH (u:User {userId: $userId})
            SET u.food_embedding = $food_emb,
                u.travel_embedding = $travel_emb,
                u.embedding_updated_at = timestamp()
            """
            session.run(store_query, userId=int(user_id), food_emb=food_embedding, travel_emb=travel_embedding)
            logging.info(f"User food and travel embeddings stored successfully for user {user_id}.")

            # Return both embeddings and summaries. Ensures a consistent 6-value return.
            return food_embedding, travel_embedding, last_known_location, visited_cities, food_summary, travel_summary

    except Exception as e:
        logging.error(f"An error occurred in get_or_create_user_embedding for user {user_id}: {e}", exc_info=True)
        # Return a default empty profile to prevent crashes downstream
        empty_emb = embedding_model.embed_query("")
        return empty_emb, empty_emb, None, [], "", ""
    finally:
        if driver:
            driver.close()