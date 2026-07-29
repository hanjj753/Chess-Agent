# Chess Agent Learning Project

This project is set up for pair-programming a chess agent without hiding the
important ideas behind a finished blob of code.

## How we will work

1. Codex writes the boring scaffolding: package layout, CLI, tests, and safe
   interfaces.
2. You read and edit the small core functions.
3. Codex reviews your changes, explains mistakes, and adds tests around the
   ideas you just implemented.

The current baseline is:

- `python-chess` for legal chess rules
- random agent as a sanity check
- negamax + alpha-beta pruning agent
- simple material evaluation
- simple move ordering

Later, this search agent can become a sparring partner or data generator for a
reinforcement-learning agent.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run a Game

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black random --depth 3
```

## Run Tests

```powershell
.\.venv\Scripts\python -m pytest
```

## Learning Checkpoints

1. Read `chess_agent/engine/search.py`.
   Focus on why the recursive score is negated after every move.

2. Modify `chess_agent/engine/evaluation.py`.
   Add one evaluation feature at a time, then run tests or games.

3. Modify `chess_agent/engine/move_ordering.py`.
   Better move ordering does not change the final answer at a fixed depth, but
   it helps alpha-beta search reach deeper positions faster.

4. Add tests before adding cleverness.
   Chess engines become confusing fast; tests keep the board from turning into
   folklore.
