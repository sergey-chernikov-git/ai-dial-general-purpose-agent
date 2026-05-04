from typing import Any, Optional

from aidial_sdk.chat_completion import Message, Role
from pydantic import StrictStr, BaseModel
from pydantic.v1 import Required, Field
from sympy import re

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams
from requests import post
import json, re


class ImageGenerationToolProperties(BaseModel):
    prompt: str = Field(description="Extensive description of the image that should be generated.")
    size: Optional[str] = Field(
        description="The size of the image to be generated.",
        default="1024x1024",
        enum=["1024x1024", "1024x1792", "1792x1024"]
    ),
    style: Optional[str] = Field(
        description="The style of the image to be generated.",
        default="vivid",
        enum=["natural", "vivid"],
    ),
    quality: Optional[str] = Field(
        description="The quality of the image to be generated.",
        default="standard",
        enum=["standard", "hd"]
    ),


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        msg = await super()._execute(tool_call_params)

        if msg.custom_content and msg.custom_content.attachments:
            for attachment in msg.custom_content.attachments:
                if attachment.type in ("image/png", "image/jpeg"):
                    tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")

            if not msg.content:
                msg.content = StrictStr('The image has been successfully generated according to request and shown to user!')

        return msg

    @property
    def deployment_name(self) -> str:
        return "gpt-image-1-mini-2025-10-06"

    @property
    def name(self) -> str:
        return "image_generator_tool"

    @property
    def description(self) -> str:
        return "# Image generator\nGenerates image based on the provided description.\n## Instructions:\n- Use that tool when user asks to generate an image based on the description or to visualize some text or information.\n- Choose the best size from available options based on user request or image type. For specific size requests, use the closest supported option.\n- When the tool returns a markdown image URL, always include it in your response and follow it with a brief description.\n## Restrictions:\n- Never use this tool for data or numerical information visualization."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated."
                },
                "size": {
                    "type": "string",
                    "description": "The size of the generated image.",
                    "enum": [
                        "1024x1024",
                        "1024x1792",
                        "1792x1024"
                    ],
                    "default": "1024x1024"
                },
                "style": {
                    "type": "string",
                    "description": "The style of the generated image. Must be one of `vivid` or `natural`. \n- `vivid` causes the model to lean towards generating hyperrealistic and dramatic images. \n- `natural` causes the model to produce more natural, less realistic looking images.",
                    "enum": [
                        "natural",
                        "vivid"
                    ],
                    "default": "natural"
                },
                "quality": {
                    "type": "string",
                    "description": "The quality of the image that will be generated. ‘hd’ creates images with finer details and greater consistency across the image.",
                    "enum": [
                        "standard",
                        "hd"
                    ],
                    "default": "standard"
                }
            },
            "required": [
                "prompt"
            ]
        }