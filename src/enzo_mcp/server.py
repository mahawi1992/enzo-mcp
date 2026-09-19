"""Enzo MCP surface: atomize, observe, and inspect state."""

from __future__ import annotations

from mcp.server import MCPServer

from .engine import EnzoEngine
from .models import (
    AtomizationResult,
    AtomizeRequest,
    InvestigationView,
    JEVResult,
    ObserveRequest,
)

engine = EnzoEngine()
mcp = MCPServer(
    "Enzo",
    version="0.3.0",
    instructions=(
        "Enzo reduces a problem into independently falsifiable atomic claims. "
        "Use enzo_atomize before enzo_observe. UNKNOWN is a valid gap, not an error."
    ),
)


@mcp.tool()
async def enzo_atomize(request: AtomizeRequest) -> AtomizationResult:
    """Validate one proposed claim and admit, decompose, or request refinement."""

    return await engine.atomize(request)


@mcp.tool()
async def enzo_observe(request: ObserveRequest) -> JEVResult:
    """Evaluate typed evidence for one admitted atomic claim.

    Deterministic evidence is authoritative. With TYPESAFE_API_KEY configured,
    semantic-only questions can be evaluated by Pydantic AI's TypeSafe/Jev provider.
    An optional local response ledger can reuse exact approved Jev responses.
    External evaluation is two-phase: first provide dispatch_selection to preview
    the exact logical request, then repeat it with allow_external_jev=true and the
    matching approved_dispatch_sha256.
    """

    return await engine.observe(request)


@mcp.tool()
async def enzo_state(investigation_id: str) -> InvestigationView:
    """Return canonical history and computed investigation status."""

    return await engine.state(investigation_id)


def main() -> None:
    """Run Enzo over stdio."""

    mcp.run()


if __name__ == "__main__":
    main()
