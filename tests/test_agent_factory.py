import pytest

from chess_agent.agent_factory import make_agent
from chess_agent.agents import AlphaBetaAgent, RandomAgent


def test_make_agent_creates_random_agent() -> None:
    assert isinstance(make_agent("random", depth=1), RandomAgent)


def test_make_agent_creates_alpha_agent_with_depth() -> None:
    agent = make_agent("alpha", depth=4)

    assert isinstance(agent, AlphaBetaAgent)
    assert agent.depth == 4


def test_make_agent_creates_alpha_agent_with_time_limit() -> None:
    agent = make_agent("alpha", depth=8, time_limit=0.5)

    assert isinstance(agent, AlphaBetaAgent)
    assert agent.time_limit == 0.5


def test_make_uci_agent_requires_engine_path() -> None:
    with pytest.raises(ValueError):
        make_agent("uci", depth=1)
