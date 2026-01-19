from backend.services.pkg_population import llm_kg_builder_cypher
from backend.services.img_description import get_img_description
from backend.services.constants import (
    QUERY_TO_RESOLVE_IMG_NAME,
    TMP_DIR,
    IMG_ANALYSIS_VL_CONFIG,
)


async def imgs_temp_path_resolver(photo_ids: list[str], user_id: int) -> list[str]:
    """
    resolves images paths in temp dir
    e.g. [photo_2342, photo_2421] --> [./data/tmp/user-1/family.jpg, ./data/tmp/user-1/dog.png]
    """
    database = f"user-{user_id}"
    img_temp_paths = []
    for p_id in photo_ids:
        img_name = await llm_kg_builder_cypher(
            QUERY_TO_RESOLVE_IMG_NAME,
            params={"photo_id": p_id},
            database=database,
        )
        img_name = img_name["data"][0]["c.fileName"].strip()
        img_temp_paths.append(f"{TMP_DIR}/{database}/{img_name}")

    return img_temp_paths


async def img_analyzer(question: str, photo_ids: list, user_id: int) -> str:
    IMG_DEEP_ANALYSIS_PROMPT = f"""
    QUESTION: {question}
    Given this question analyze the image and, if it's possible, return a clear answer (Discard temporal information). 
    If not, only respond "I can't respond to this question.".
    """

    imgs_temp_path = await imgs_temp_path_resolver(photo_ids, user_id)

    descriptions = []
    for img_path in imgs_temp_path:
        img_description = get_img_description(
            question=IMG_DEEP_ANALYSIS_PROMPT,
            img_path=img_path,
            VL_model_env_value=IMG_ANALYSIS_VL_CONFIG,
        )
        if not img_description == "I can't respond to this question.":
            descriptions.append(img_description)

    if descriptions:
        joined_desc = "\n- ".join(descriptions)
        final_response = f"Answers by analyzing different pictures: \n- {joined_desc}"
    else:
        final_response = "I can't respond to this question."
    return final_response


# -----------------------------------------------------------------------------
# Development/Testing Entry Point
# -----------------------------------------------------------------------------
# This section is for local development testing only.
# Run with: python -m backend.services.analyzer

if __name__ == "__main__":
    import asyncio

    async def _test_analyzer():
        """Test function for local development."""
        res = await img_analyzer("What color is the laundry basket at my house?", [], 1)
        print(res)

    asyncio.run(_test_analyzer())
