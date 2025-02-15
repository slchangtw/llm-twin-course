import opik
from langchain_ollama import ChatOllama
from opik.integrations.langchain import OpikTracer

from core.config import settings
from core.lib import remove_think_tags
from core.rag.prompt_templates import QueryExpansionTemplate


class QueryExpansion:
    opik_tracer = OpikTracer(tags=["QueryExpansion"])

    @staticmethod
    @opik.track(name="QueryExpansion.generate_response")
    def generate_response(query: str, to_expand_to_n: int) -> list[str]:
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(to_expand_to_n)
        model = ChatOllama(
            model=settings.CHAT_MODEL_ID,
            temperature=0,
        )
        chain = prompt | model
        chain = chain.with_config({"callbacks": [QueryExpansion.opik_tracer]})

        response = chain.invoke({"question": query})
        response = response.content

        queries = [
            q.strip()
            for q in remove_think_tags(response).split(
                query_expansion_template.separator
            )
            if q.strip()
        ]
        stripped_queries = [
            stripped_item for item in queries if (stripped_item := item.strip(" \\n"))
        ]

        return stripped_queries
