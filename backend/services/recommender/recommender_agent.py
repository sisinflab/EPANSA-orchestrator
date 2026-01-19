import logging
from .user_profile_builder import get_or_create_user_embedding
from .intent_extractor import extract_intent_from_question
from .poi_recommender import POIRecommender, geocode_location, _find_best_city_match
from libs.llm_graph_builder.src.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


RESPONSE_GENERATION_PROMPT_TEMPLATE = """
You are an AI assistant for POI recommendations.
Based on user request and a list of top matching POIs, provide a helpful and conversational answer.
Mention the top options (up to 5) from the following list and briefly explain why they might be a good fit for the user.

User's question: "{user_question}"

Top Recommendations (Name, Category, Description):
{top_recommendations_list_str}

Your response:
"""

# fallback used to generate a friendly response when the system encounters an error.
SYSTEM_RESPONSE_PROMPT_TEMPLATE = """
You are a helpful and polite AI assistant.
Your user asked a question, but the system could not find a specific answer.
Based on the user's question and the system message, provide a friendly and helpful response in the user's original language.

User's original question: "{user_question}"
System message to convey: "{system_message}"

Your helpful response:
"""

GENERAL_KNOWLEDGE_FALLBACK_PROMPT_TEMPLATE = """
You are a helpful and knowledgeable local guide AI.
Your internal database could not find specific venues for the user's request, but you must still provide a helpful and specific answer using your general knowledge.

The user is asking about things to do in the following location: **{city_for_recommendation}**

Based on the user's question, provide a list of IDEAS or TYPES of places they could visit in that specific city. Be creative and inspiring.
For example, if the user asks for "restaurants in Rome", suggest types of restaurants (trattoria, osteria), famous food districts (Trastevere), or specific dishes to try.
Do not invent specific venue names unless they are world-famous landmarks (e.g., Colosseum). Focus on categories, areas, and experiences relevant to the specified city.

User's question: "{user_question}"

Your helpful, general knowledge-based response for **{city_for_recommendation}**:
"""

# A set of POI categories that are considered related to food or nightlife.
FOOD_NIGHTLIFE_CATEGORIES = {"restaurant", "nightlife"}


def extract_city_from_address(address_string: str) -> str | None:
    """
    Extracts the city name from a full address string for comparison purposes.
    """
    if not isinstance(address_string, str):
        return None
    parts = [p.strip() for p in address_string.split(",")]
    if not parts:
        return None
    # city is the second to last part
    return parts[-2] if len(parts) >= 2 else parts[0]


class RecommenderAgent:
    """
    Orchestrates the entire recommendation process, from understanding user intent
    to generating a final, conversational response. It manages user profiles with
    separate embeddings for different interest domains (food vs. travel).
    """

    def __init__(
        self,
        database: str,
        user_id: int,
        embedding_model,
        model_env_value: str,
        poi_data_path: str,
    ):
        self.database = database
        self.user_id = user_id
        self.embedding_model = embedding_model
        self.model_env_value = model_env_value
        self.recommender = POIRecommender(poi_data_path)

        self.last_known_location_str = None
        self.user_food_embedding = None
        self.user_travel_embedding = None
        self.visited_cities = []
        self.user_food_summary = ""
        self.user_travel_summary = ""

    async def initialize(self):
        """
        Loads the user's dual profiles (food and travel) from the database,
        creating them if they don't exist. This populates all user-specific
        attributes of the agent instance.
        """
        (
            self.user_food_embedding,
            self.user_travel_embedding,
            self.last_known_location_str,
            self.visited_cities,
            self.user_food_summary,
            self.user_travel_summary,
        ) = await get_or_create_user_embedding(
            self.database, self.user_id, self.embedding_model, self.model_env_value
        )
        if not self.user_food_embedding or not self.user_travel_embedding:
            logging.error(
                "Failed to initialize RecommenderAgent: User embeddings could not be loaded."
            )
            raise ValueError("User embeddings are not available.")
        if self.last_known_location_str:
            logging.info(
                f"Last known location for user {self.user_id} is '{self.last_known_location_str}'"
            )

    async def _generate_final_response(self, user_question: str, system_message: str):
        """Helper function to generate a final, localized response using an LLM in case of errors."""
        llm, _ = get_llm(self.model_env_value)
        prompt = PromptTemplate.from_template(SYSTEM_RESPONSE_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        response = await chain.ainvoke(
            {"user_question": user_question, "system_message": system_message}
        )
        return response

    async def get_response(self, user_question: str):
        """
        Main orchestration method. It understands the user's request, selects the appropriate
        user profile, gets recommendations, and formats a natural language response.
        """
        if not self.user_food_embedding or not self.user_travel_embedding:
            system_msg = "The system could not load the user's profile."
            return await self._generate_final_response(user_question, system_msg)

        intent = await extract_intent_from_question(user_question, self.model_env_value)
        if not intent:
            system_msg = "The system could not understand the user's request."
            return await self._generate_final_response(user_question, system_msg)

        poi_category = intent.get("poi_category")

        if poi_category == "destination":
            logging.info(
                "> Destination request detected. Using new LLM re-ranking flow."
            )

            # list of cities visited by the user from their profile.
            all_cities_to_exclude_raw = list(self.visited_cities)

            # cities the user wants to exclude from their current question.
            explicit_exclusions = intent.get("exclude_locations")
            if explicit_exclusions and isinstance(explicit_exclusions, list):
                logging.info(
                    f"User explicitly requested to exclude: {explicit_exclusions}"
                )
                all_cities_to_exclude_raw.extend(explicit_exclusions)

            # normalize the entire raw list using fuzzy matching to get canonical names.
            canonical_cities_to_exclude = set()
            if all_cities_to_exclude_raw:
                logging.info(
                    f"Normalizing full exclusion list: {list(set(all_cities_to_exclude_raw))}"
                )
                available_cities_for_matching = self.recommender.available_cities
                for city_to_exclude in set(all_cities_to_exclude_raw):
                    matched_city = _find_best_city_match(
                        city_to_exclude.lower(), available_cities_for_matching
                    )
                    if matched_city:
                        logging.info(
                            f"-> Fuzzy matching '{city_to_exclude}' to canonical name '{matched_city}' for exclusion."
                        )
                        canonical_cities_to_exclude.add(matched_city)

            # Get a large pool of candidates ranked by theme match
            candidate_destinations_df = (
                await self.recommender.get_destination_recommendations(
                    user_profile_vector=self.user_travel_embedding,
                    topk=20,
                    exclude_cities=list(canonical_cities_to_exclude),
                    intent=intent,
                    # Pass user location for practicality scoring
                    home_base_str=self.last_known_location_str,
                )
            )

            if candidate_destinations_df.empty:
                system_msg = (
                    "I couldn't find any new destinations matching your criteria."
                )
                return await self._generate_final_response(user_question, system_msg)

            # Use LLM to re-rank the candidates based on practical constraints
            logging.info(
                f"Sending {len(candidate_destinations_df)} candidates to LLM for pragmatic re-ranking..."
            )
            final_recs = await self.recommender.rerank_destinations_with_llm(
                candidates_df=candidate_destinations_df,
                user_profile_summary=self.user_travel_summary,
                intent=intent,
                home_base_str=self.last_known_location_str,
                model_env_value=self.model_env_value,
                user_profile_vector=self.user_travel_embedding,
            )

            if not final_recs:
                system_msg = "The system could not find any new destination recommendations matching your specific request."
                return await self._generate_final_response(user_question, system_msg)

            # Format the final response based on the LLM's output
            response_parts = [
                "Based on your request, here are a few travel ideas that should be a great fit:"
            ]
            for r in final_recs[:3]:
                pois_list = "\n".join(
                    [
                        f"     - {p.get('name')} ({p.get('primary_category')})"
                        for p in r.get("pois", [])[:10]
                    ]
                )
                response_parts.append(
                    f"\n- **{r['city'].upper()}, {r.get('country', '')}**\n"
                    f"   *Why it's a good fit:* {r['justification']}\n"
                    f"   *Not to be missed:*\n{pois_list}"
                )
            return "\n".join(response_parts)

        # Specific POI Recommendation
        else:
            if poi_category in FOOD_NIGHTLIFE_CATEGORIES:
                logging.info(
                    f"> POI request for '{poi_category}'. Using FOOD embedding."
                )
                active_embedding = self.user_food_embedding
                active_summary = self.user_food_summary
            else:
                logging.info(
                    f"> POI request for '{poi_category or 'general'}'. Using TRAVEL embedding."
                )
                active_embedding = self.user_travel_embedding
                active_summary = self.user_travel_summary

            # Determine the city for the recommendation.
            # If the intent contains a location, use it directly.
            if intent.get("location"):
                city_for_recommendation_str = intent.get("location")
            # Otherwise, use the last known location
            else:
                city_for_recommendation_str = extract_city_from_address(
                    self.last_known_location_str
                )

            if not city_for_recommendation_str:
                system_msg = "I'm not sure which city you're asking about. Please specify a location."
                return await self._generate_final_response(user_question, system_msg)

            # Now, city_for_recommendation_str is either the user's input or a clean "City, Country" string.
            logging.info(
                f"City for recommendation has been set to: '{city_for_recommendation_str}'"
            )

            # Decide whether to use distance in the ranking logic.
            city_from_intent = extract_city_from_address(intent.get("location"))
            city_from_profile = extract_city_from_address(self.last_known_location_str)

            use_distance_scoring = (city_from_intent is None) or (
                city_from_intent
                and city_from_profile
                and city_from_intent.lower() == city_from_profile.lower()
            )

            user_coords = (
                geocode_location(city_for_recommendation_str)
                if use_distance_scoring
                else None
            )

            if use_distance_scoring:
                logging.info(
                    f"Distance scoring ENABLED: User is asking for local recommendations in '{city_for_recommendation_str}'."
                )
            else:
                logging.info(
                    f"Distance scoring DISABLED: User is planning for a different city ('{city_for_recommendation_str}')."
                )

            recs = await self.recommender.get_recommendations(
                user_profile_summary=active_summary,
                model_env_value=self.model_env_value,
                intent=intent,
                user_profile_vector=active_embedding,
                user_location=user_coords,
                city_name=city_for_recommendation_str,
            )
            if not recs:
                logging.warning(
                    "Internal recommender returned no results. Falling back to general LLM knowledge."
                )

                llm, _ = get_llm(self.model_env_value)
                prompt = ChatPromptTemplate.from_template(
                    GENERAL_KNOWLEDGE_FALLBACK_PROMPT_TEMPLATE
                )
                chain = prompt | llm | StrOutputParser()

                # Pass the city context to the fallback prompt
                final_response = await chain.ainvoke(
                    {
                        "user_question": user_question,
                        "city_for_recommendation": city_for_recommendation_str,
                    }
                )
                return final_response

            # Format the final response for the user.
            recommendations_list_str = []
            for rec in recs:
                rec_str = (
                    f"- Name: {rec['name']}\n  Category: {rec['primary_category']}\n"
                )
                rec_str += f"  Why you might like it: {rec.get('why', 'It seems like a good match for you.')}"
                recommendations_list_str.append(rec_str)

            recommendations_for_prompt = "\n".join(recommendations_list_str)
            llm, _ = get_llm(self.model_env_value)
            prompt = ChatPromptTemplate.from_template(
                RESPONSE_GENERATION_PROMPT_TEMPLATE
            )
            chain = prompt | llm | StrOutputParser()

            final_response = await chain.ainvoke(
                {
                    "user_question": user_question,
                    "top_recommendations_list_str": recommendations_for_prompt,
                }
            )

            return final_response
