import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You are a professional that answers questions with a document context.

As user input you will receive:
- CONTEXT: Document data to use in the final answer (document context)
- REQUEST: The personal user question, that should be answered with the CONTEXT information.

Output Limits:
Please answer the question based only on the provided CONTEXT. 
If the CONTEXT does not contain relevant information to answer the question, respond with "I don't know". 
Do not use any information that is not included in the CONTEXT. Be concise and to the point in your answer.
"""


class RagToolParameters(BaseModel):
    request: str = Field(description="The search query or question to search for in the document")
    file_url: str


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache, device: str = 'cpu'):
        self._endpoint = endpoint
        self._deployment_name = deployment_name
        self._document_cache = document_cache
        self._model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2", device=device)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_tool"

    @property
    def description(self) -> str:
        return """
        The tool is able to make search in any document that can contains text.
        Used only when a textx search in a large document required.
        """

    @property
    def parameters(self) -> dict[str, Any]:
        return RagToolParameters.model_json_schema()

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments.get('request')
        file_url = arguments.get('file_url')
        stage = tool_call_params.stage
        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")
        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        document_cached = self._document_cache.get(cache_document_key)
        if document_cached:
            index, chunks = document_cached
        else:
            text = DialFileContentExtractor(
                endpoint=self.endpoint,
                api_key=tool_call_params.api_key
            ).extract_text(file_url)

            if not text:
                stage.append_content("## Response: \n")
                content = "Error: File content not found."
                stage.append_content(f"{content}\n")
                return content

            chunks = self._text_splitter.split_text(text)
            embeddings = self._model.encode(chunks)
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype('float32'))
            self._document_cache.set(cache_document_key, index, chunks)

        query_embedding = self.model.encode([request]).astype('float32')
        k = min(3, len(chunks))
        distances, indices = index.search(query_embedding, k=k)
        retrieved_chunks = []
        for idx in indices[0]:
            retrieved_chunks.append(idx)
        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content(f"## Response: \n")

        dial = AsyncDial(
            api_version="025-01-01-preview",
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
        )

        messages = [
            {
                "role": Role.SYSTEM,
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": Role.USER,
                "content": augmented_prompt,
            }
        ]

        chunk_stream = dial.chat.completions.create(
            messages,
            deployment_name=self._deployment_name,
            stream=True
        )
        collected_content = ""
        async for chunk in chunk_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    tool_call_params.stage.append_content(delta.content)
                    collected_content += delta.content
        return collected_content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        chunks_line = "\n\n".join(chunks)
        return f"CONTEXT:\n{chunks_line}\n---\nREQUEST: {request}"
