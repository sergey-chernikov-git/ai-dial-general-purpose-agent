import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        self._endpoint = endpoint
        self._system_prompt = system_prompt
        self._tools = tools
        self._tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self._state = {
            "TOOL_CALL_HISTORY_KEY": []
        }

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request,
                             response: Response) -> Message:
        dial_async = AsyncDial(base_url=self._endpoint, api_key=request.api_key, api_version=request.api_version)
        chunks = dial_async.chat.completions.create(
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self._tools],
            deployment_name=deployment_name,
            stream=True
        )
        tool_call_index_map = {}
        content = ""
        custom_content: CustomContent = CustomContent(attachments=[])
        async for chunk in chunks:
            if chunk.get('choices'):
                delta = chunk.get('choices')[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content
                if delta and delta.tool_calls:
                    for call in delta.tool_calls:
                        if call.id:
                            tool_call_index_map[call.index] = call
                        else:
                            tool_call = tool_call_index_map[call.index]
                            if call.function:
                                argument_chunk = call.function.arguments or ''
                                tool_call.function.arguments += argument_chunk
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=custom_content,
            tool_calls=[ToolCall.model_validate(tool_call) for tool_call in tool_call_index_map.values()]
        )

        if assistant_message.tool_calls:
            tasks = [
                self._process_tool_call(
                    tool_call=tool_call,
                    choice=choice,
                    api_key=request.api_key,
                    conversation_id=request.headers['x-conversation-id']
                )
                for tool_call in assistant_message.tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)

            self._state[TOOL_CALL_HISTORY_KEY].append(assistant_message.dict(exclude_none=True))
            self._state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            return await self.handle_request(
                deployment_name=deployment_name,
                choice=choice,
                request=request,
                response=response
            )

        choice.set_state(self._state)

        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        unpacked_msgs = unpack_messages(messages, self._state[TOOL_CALL_HISTORY_KEY])
        unpacked_msgs.insert(0,  {
                "role": Role.SYSTEM.value,
                "content": self._system_prompt,
        })
        print("================ History ====================")
        for msg in unpacked_msgs:
            print(f"     {json.dumps(msg)}")

        return unpacked_msgs

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[
        str, Any]:
        name = tool_call.function.name
        stage = StageProcessor.open_stage(
            choice=choice,
            name=name
        )
        tool = self._tools_dict.get(name)
        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r")
            stage.append_content("## Response: \n")

        tool_message = await tool.execute(
            ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id
            )
        )

        StageProcessor.close_stage_safely(stage)

        return tool_message.dict(exclude_none=True)
