from typing import Any

from chess_agent.agents import AlphaBetaAgent, HumanAgent, RandomAgent, UciEngineAgent
from chess_agent.agents.base import Agent

AGENT_CHOICES = ["random", "alpha", "human", "uci"]


def make_agent(
    kind: str,
    *,
    depth: int,
    engine_path: str | None = None,
    engine_time: float | None = None,
    engine_depth: int | None = None,
    engine_nodes: int | None = None,
    engine_options: dict[str, Any] | None = None,
) -> Agent:
    if kind == "random":
        return RandomAgent()
    if kind == "alpha":
        return AlphaBetaAgent(depth=depth)
    if kind == "human":
        return HumanAgent()
    if kind == "uci":
        if engine_path is None:
            raise ValueError("uci agent requires an engine path")
        return UciEngineAgent(
            engine_path,
            time_limit=engine_time,
            depth=engine_depth,
            nodes=engine_nodes,
            options=engine_options,
        )
    raise ValueError(f"unknown agent: {kind}")


def close_agent(agent: Agent) -> None:
    close = getattr(agent, "close", None)
    if close is not None:
        close()
