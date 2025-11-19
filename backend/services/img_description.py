import logging
import base64
from groq import Groq

logging.basicConfig(level="INFO")


def get_img_description(question: str, img_path: str, VL_model_env_value: str):
    """
    Generates image description using a chain of experts: BLIP2, LLaVA, and Qwen2-VL.
    """
    try:
        logging.info(f"loading the image...")
        with open(img_path, "rb") as file:
            base64_img = base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error loading images: {str(e)}")

    try:
        _, model_name, _, api_key = VL_model_env_value.split(",")
        vlm = Groq(api_key=api_key)
        desc = vlm.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{question}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            stream=False,
            stop=None,
        )
        desc = desc.choices[0].message.content
    except Exception as e:
        logging.error(f"Error processing descriptions: {str(e)}")
        desc = ""

    return desc
