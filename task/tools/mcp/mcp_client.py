from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        print(f"mcp_server_url: {mcp_server_url}")
        self._mcp_server_url = mcp_server_url
        self._session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        print(f"mcp_server_url: {mcp_server_url}")
        client = cls(mcp_server_url)
        await client.connect()
        return client

    async def connect(self):
        """Connect to MCP server"""
        if self._session:
            return
        print(f"mcp_server_url: {self._mcp_server_url}")
        self._streams_context = streamablehttp_client(self._mcp_server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self._session: ClientSession = await self._session_context.__aenter__()

        await self._session.initialize()
        try:
            await self._session.send_ping()
        except Exception as e:
            # Clean up on connection failure
            await self.close()
            raise ValueError(f"MCP server connection failed: {e}")

    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        tools = (await self._session.list_tools()).tools
        return [
            MCPToolModel(name=tool.name, description=tool.description, parameters=tool.outputSchema)
            for tool in tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        result: CallToolResult = await self._session.call_tool(name=tool_name, arguments=tool_args)
        return result.content[0]

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""

        resource = await self._session.read_resource(uri)
        content = resource.contents[0]

        if isinstance(content, TextResourceContents):
            return content.text

        if isinstance(content, BlobResourceContents):
            return content.blob

        return ""

    async def close(self):
        """Close connection to MCP server"""
        self._session_context.close()
        self._streams_context.close()
        self._session_context = None
        self._streams_context = None
        self._session = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False
