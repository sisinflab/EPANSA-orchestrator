import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from libs.llm_graph_builder.src.llm import get_llm

# prompt
INTENT_EXTRACTION_PROMPT_TEMPLATE = """
From the user's question, extract key entities for a POI or destination recommendation.
Return ONLY a valid JSON object adhering to the specified schema. DO NOT include any text or markdown before or after the JSON object.

JSON Schema:
{{
  "poi_category": "string (e.g., 'restaurant', 'museum', 'destination')",
  "subcategory": "string (e.g., 'seafood', 'cathedral') or null",
  "location": "string (City or area, e.g., 'Milan') or null",
  "country": "string (e.g., 'Italy', 'Spain') or null",
  "continent": "string ('Europe', 'North America', 'South America', 'Asia', 'Africa', 'Oceania') or null",
  "duration": "string ('short' for trips of 3 days or less, 'long' for longer trips) or null",
  "exclude_location": "string or list of strings (Cities to exclude)",
  "exclude_country": "string or list of strings (Countries to exclude)",
  "exclude_continent": "string or list of strings (Continents to exclude)"
}}

Rules:
1.  **Local vs. Destination:** This is the most important distinction.
    *   A query is for a **"destination"** ONLY if the user is asking about planning a trip, vacation, or journey to a place they are NOT currently in. Keywords like "trip", "vacation", "holiday", "where should I travel" are strong indicators.
    *   A query is **LOCAL** if it implies immediate action or proximity. Time-related words like **"tonight", "today", "now"** STRONGLY indicate a local search, NOT a destination search. For example, "Where to go tonight?" is a local request for `nightlife`, NOT a `destination` search.

2.  **POI in a Specific City:** If the user asks for a specific type of place in a specific city (e.g., "restaurants in Milan"), it is a POI search. Set `poi_category` to "restaurant" and `location` to "Milan". It is NOT a "destination" search.

3.  **General Sightseeing:** If the user asks general "what to see" or "things to do" in a city, set `poi_category` to "attractions".

4.  **Default to Local:** If a query is ambiguous but implies immediacy (like "I'm hungry" or "I'm bored"), assume it's a local search for "restaurant" or "attractions". DO NOT default to "destination".

5.  **Continents:** Be specific about continents. If the user says "America", consider the context. If ambiguous, set to null. If they say "South America", set it to "South America".

User question: "{user_question}"
"""


# JSON parsing helper
def _extract_and_parse_json(text: str) -> dict:
    """
    Extracts a JSON object from a string and parses it.
    Handles cases where the JSON is embedded in other text.
    """
    if not isinstance(text, str):
        return {}

    # Find the balanced JSON object
    start_brace = text.find("{")
    if start_brace == -1:
        return {}

    end_brace = -1
    brace_count = 0
    for i, char in enumerate(text[start_brace:]):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        if brace_count == 0:
            end_brace = start_brace + i + 1
            break

    if end_brace == -1:
        return {}

    json_str = text[start_brace:end_brace]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse extracted JSON string: {json_str}")
        return {}


async def extract_intent_from_question(
    user_question: str, model_env_value: str = "OPENAI_GPT4O_MINI"
) -> dict:
    """
    Extract an intent JSON from the user's question using an LLM with robust JSON parsing.
    """
    try:
        llm, _ = get_llm(model_env_value)
        prompt = ChatPromptTemplate.from_template(INTENT_EXTRACTION_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        response_str = await chain.ainvoke({"user_question": user_question})
        # +++ USE NEW ROBUST PARSER +++
        intent_json = _extract_and_parse_json(response_str)
        if not intent_json:
            logging.error(
                f"Could not extract valid JSON from LLM response: {response_str}"
            )
            return {}

        uq_lower = user_question.lower()

        destination_keywords = [
            "destination",
            "vacation",
            "trip",
            "travel",
            "journey",
            "holiday",
            "weekend",
        ]

        poi_category = intent_json.get("poi_category", "").lower()

        # If a specific location is mentioned, it is NOT a destination search, unless it's a sightseeing query.
        if intent_json.get("location"):
            is_sightseeing_query = any(
                kw in uq_lower
                for kw in [
                    "what to see",
                    "what to visit",
                    "things to do",
                    "attractions",
                ]
            )
            if is_sightseeing_query:
                intent_json["poi_category"] = "attractions"
            elif any(keyword in poi_category for keyword in destination_keywords):
                intent_json["poi_category"] = "attractions"
        else:
            # If NO location is mentioned, check for travel-related keywords to force "destination" category.
            if any(keyword in uq_lower for keyword in destination_keywords) or any(
                keyword in poi_category for keyword in destination_keywords
            ):
                intent_json["poi_category"] = "destination"
                logging.info(
                    "Normalized query to 'destination' based on travel keywords."
                )

        # Map common fine-grained words into macro categories (light-touch)
        cat_map = {
            "architecture": [
                "architecture",
                "castle",
                "church",
                "cathedral",
                "monument",
                "historic site",
                "landmark",
            ],
            "restaurant": ["restaurant", "food", "dining", "eatery", "cafe"],
            "nightlife": ["bar", "pub", "club", "cocktail", "nightlife"],
            "museum": ["museum", "gallery", "art", "exhibition"],
            "nature": ["park", "garden", "beach", "mountain"],
            "shopping": ["mall", "store", "market"],
        }
        final_poi_cat = (intent_json.get("poi_category") or "").lower()
        for macro, terms in cat_map.items():
            if any(term in final_poi_cat for term in terms):
                intent_json["poi_category"] = macro
                break

        logging.info(f"Final Extracted Intent: {intent_json}")
        return intent_json

    except Exception as e:
        logging.error(f"An error occurred during intent extraction: {e}", exc_info=True)
        return {}
