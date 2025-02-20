import json

from langchain_ollama import ChatOllama

from core import get_logger
from core.config import settings
from core.lib import remove_json_syntax_highlighting, remove_think_tags

MAX_LENGTH = 16384
SYSTEM_PROMPT = (
    "You are a technical writer handing someone's account to post about AI and MLOps."
)


logger = get_logger(__name__)


class GptCommunicator:
    def __init__(self, gpt_model: str = settings.MODEL_ID):
        self.gpt_model = gpt_model

    def send_prompt(self, prompt: str) -> list:
        try:
            client = ChatOllama(model=settings.MODEL_ID)
            logger.info(f"Sending batch to GPT = '{settings.MODEL_ID}'.")

            message = [
                ("system", SYSTEM_PROMPT),
                ("user", prompt[:MAX_LENGTH]),
            ]

            chat_completion = client.invoke(message)
            response = chat_completion.content
            processed_response = remove_think_tags(response)
            processed_response = remove_json_syntax_highlighting(processed_response)
            return json.loads(processed_response)
        except Exception:
            logger.exception(
                f"Skipping batch! An error occurred while communicating with API."
            )

            return []
