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
            TOOL_CALL_HISTORY_KEY: []
        }

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request,
                             response: Response) -> Message:
        print("GeneralPurposeAgent -> handle_request() -> start ")
        api_key = request.api_key

        client: AsyncDial = AsyncDial(
            base_url=self._endpoint,
            api_key=api_key,
            api_version=request.api_version,
        )
        tools = [tool.schema for tool in self._tools]
        print(f"GeneralPurposeAgent -> handle_request() -> tools initialized {str(tools)}")
        messages = self._prepare_messages(request.messages)
        print("GeneralPurposeAgent -> handle_request() -> messages initialized ")
        print("GeneralPurposeAgent -> handle_request() -> client.chat.completions.create started")
        chunks = await client.chat.completions.create(
            messages=messages,
            tools=tools,
            stream=True,
            deployment_name=deployment_name,
        )
        print("GeneralPurposeAgent -> handle_request() -> client.chat.completions.create finished ")
        tool_call_index_map = {}
        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        count = 0
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.id:
                            tool_call_index_map[tool_call_delta.index] = tool_call_delta
                        else:
                            tool_call = tool_call_index_map[tool_call_delta.index]
                            if tool_call_delta.function:
                                argument_chunk = tool_call_delta.function.arguments or ''
                                tool_call.function.arguments += argument_chunk

        print("GeneralPurposeAgent -> handle_request() ->  async for chunk in chunks ")
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content,
            custom_content=custom_content,
            tool_calls=[ToolCall.validate(tool_call) for tool_call in tool_call_index_map.values()]
        )

        if assistant_message.tool_calls:
            print(f"Calling tools: {assistant_message.tool_calls}")
            tasks = [
                self._process_tool_call(
                    tool_call=tool_call,
                    choice=choice,
                    api_key=api_key,
                    conversation_id=request.headers['x-conversation-id']
                )
                for tool_call in assistant_message.tool_calls
            ]
            tool_messages = await asyncio.gather(*tasks)
            print("GeneralPurposeAgent -> handle_request() -> tool_messages: ", tool_messages)
            self._state[TOOL_CALL_HISTORY_KEY].append(assistant_message.dict(exclude_none=True))
            self._state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)
            print("GeneralPurposeAgent -> handle_request() -> next ")
            return await self.handle_request(
                deployment_name=deployment_name,
                choice=choice,
                request=request,
                response=response
            )
        print("GeneralPurposeAgent -> handle_request() -> finish ")
        choice.set_state(self._state)
        print("GeneralPurposeAgent -> handle_request() -> assistant_message: ", assistant_message)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        print("GeneralPurposeAgent -> _prepare_messages() -> start")
        unpacked_msgs = unpack_messages(messages, self._state[TOOL_CALL_HISTORY_KEY])
        unpacked_msgs.insert(0, {
            "role": Role.SYSTEM.value,
            "content": self._system_prompt,
        })
        print("================ History ====================")
        try:
            for msg in unpacked_msgs:
                print(f"     {json.dumps(msg)}")
        except Exception as e:
            print(f"History error: {e}")
        print("=============================================")
        print("GeneralPurposeAgent -> _prepare_messages() -> finish")
        return unpacked_msgs

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[str, Any]:
        print("GeneralPurposeAgent -> _process_tool_call() -> start")
        name = tool_call.function.name
        stage = StageProcessor.open_stage(
            choice=choice,
            name=name
        )
        tool = self._tools_dict.get(name)
        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            try:
                stage.append_content(
                    f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r")
            except Exception as e:
                print("That fucking error when parsing tool arguments: ", e)
                print(tool_call.function.arguments)
                stage.append_content(f"```\n\r{tool_call.function.arguments}\n\r```\n\r")
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
        print("GeneralPurposeAgent -> _process_tool_call() -> finish")
        return tool_message.dict(exclude_none=True)
