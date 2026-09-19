from __future__ import annotations

from mcp import Client

from enzo_mcp.server import mcp


async def test_server_advertises_only_the_three_enzo_tools() -> None:
    async with Client(mcp) as client:
        result = await client.list_tools()

    assert {tool.name for tool in result.tools} == {
        "enzo_atomize",
        "enzo_observe",
        "enzo_state",
    }
