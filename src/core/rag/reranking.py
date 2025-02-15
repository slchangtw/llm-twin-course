from langchain_ollama import ChatOllama

from core.config import settings
from core.lib import remove_think_tags
from core.rag.prompt_templates import RerankingTemplate


class Reranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        reranking_template = RerankingTemplate()
        prompt = reranking_template.create_template(keep_top_k=keep_top_k)
        model = ChatOllama(model=settings.CHAT_MODEL_ID)
        chain = prompt | model

        stripped_passages = [
            stripped_item for item in passages if (stripped_item := item.strip())
        ]
        passages = reranking_template.separator.join(stripped_passages)
        response = chain.invoke({"question": query, "passages": passages})
        response = response.content

        reranked_passages = remove_think_tags(response).split(
            reranking_template.separator
        )
        stripped_passages = [
            stripped_item
            for item in reranked_passages
            if (stripped_item := item.strip())
        ]

        return stripped_passages
