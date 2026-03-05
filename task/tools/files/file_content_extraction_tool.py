import json
from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel
from pydantic.v1 import Required, Field

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import API_KEY, API_ENDPOINT
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionToolInput(BaseModel):
    file_url: Required[str] = Field(default=None)
    page: int = Field(default=1, description="For large documents pagination is enabled. Each page consists of 10000 characters.")


class FileContentExtractionTool(BaseTool):
    """
    Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
    PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
    USAGE: Start with page=1 (by default)
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "file_content_extractor_tool"

    @property
    def description(self) -> str:
        return """
                Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
                PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
                USAGE: Start with page=1 (by default)
                """

    @property
    def parameters(self) -> dict[str, Any]:
        return FileContentExtractionToolInput.model_json_schema()

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:

        args = json.loads(tool_call_params.tool_call.parameters)
        file_url = args.get('file_url')
        page = args.get('page') or 1
        stage = tool_call_params.stage
        stage.append_content(f"## Request arguments: \n**File URL**: {file_url}\n\r")
        if page > 1:
            stage.append_content("**Page**: {page}\n\r")
        stage.append_content("## Response: \n")
        content = DialFileContentExtractor(API_ENDPOINT, API_KEY).extract_text(file_url=file_url)
        if not content:
            raise Exception("Error: File content not found.")
        if len(content) > 10_000:
            page_size = 10_000
            total_pages = (len(content) + page_size - 1) // page_size
            if page < 1:
                page = 1
            if page > total_pages:
                content = f"Error: Page {page} does not exist. Total pages: {total_pages}"
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            page_content = page[start_index:end_index]
            content = f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"
            stage.append_content(f"```text\n\r{content}\n\r```\n\r")
        return content
