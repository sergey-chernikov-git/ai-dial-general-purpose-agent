from typing import Any, Optional

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr, BaseModel
from pydantic.v1 import Required, Field

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationToolProperties(BaseModel):
    prompt: str = Field(description="Extensive description of the image that should be generated.")
    size: str = Field(description="The size of the image to be generated.", default="1024x1024"),
    quality: str = Field(description="The quality of the image to be generated.", default="hd"),
    style: str = Field(description="The style of the image to be generated.", default="vivid"),


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        result = await super()._execute(tool_call_params)
        if result.custom_content and result.custom_content.attachments:
            [
                tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")
                for attachment in result.custom_content.attachments
                if attachment.type in ["image/png", "image/jpeg"]
            ]
            if not result.content:
                result.content = StrictStr(
                    'The image has been successfully generated according to request and shown to user!')
        return result

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generator_tool"

    @property
    def description(self) -> str:
        return "# Image generator\nGenerates image based on the provided description.\n## Instructions:\n- Use that tool when user asks to generate an image based on the description or to visualize some text or information.\n- Choose the best size from available options based on user request or image type. For specific size requests, use the closest supported option.\n- When the tool returns a markdown image URL, always include it in your response and follow it with a brief description.\n## Restrictions:\n- Never use this tool for data or numerical information visualization."

    @property
    def parameters(self) -> dict[str, Any]:
        return ImageGenerationToolProperties.model_json_schema()