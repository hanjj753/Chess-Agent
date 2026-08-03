# 체스 에이전트 학습 프로젝트

이 프로젝트는 체스를 두는 에이전트를 만들면서, 완성된 코드를 그냥 받아쓰기보다
핵심 아이디어를 직접 이해하고 바꿔보는 것을 목표로 합니다.

## 현재 기준 버전

- `python-chess`로 체스 규칙 처리
- 랜덤 에이전트
- 사람이 직접 수를 입력하는 에이전트
- 외부 UCI 엔진 실행용 에이전트
- negamax + alpha-beta pruning 기반 에이전트
- 기물 점수 + piece-square table 기반 평가 함수
- 잡는 수, 프로모션, 체크를 먼저 보는 단순 move ordering

나중에는 이 탐색 기반 에이전트를 강화학습 에이전트의 상대, 평가 기준,
데이터 생성기로 사용할 수 있습니다.

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

이미 `.venv`가 만들어져 있다면 설치 명령만 다시 실행하면 됩니다.

## 대국 실행

Alpha-beta 에이전트와 랜덤 에이전트를 대국시키려면:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black random --depth 3
```

사람이 백으로 직접 두고 alpha-beta 에이전트와 대국하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white human --black alpha --depth 3
```

입력할 수의 예시는 다음과 같습니다.

- SAN 표기: `e4`, `Nf3`, `O-O`, `Qxe5+`
- UCI 표기: `e2e4`, `g1f3`, `e7e8q`
- 종료: `quit`, `exit`, `resign`

외부 UCI 엔진과 대국하려면 엔진 실행 파일을 `engines/` 아래에 넣고 `uci` 에이전트를 사용합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine stockfish --depth 3 --engine-time 0.1
```

자세한 폴더 예시는 [engines/README.md](<D:/Workspace/Chess/engines/README.md>)를 참고하면 됩니다.

## 여러 판 평가 실행

한 판이 아니라 여러 판을 돌려 승/무/패와 점수율을 보려면 `match` 모드를 사용합니다.
기본적으로 매 게임마다 백/흑을 번갈아 둡니다.

```powershell
.\.venv\Scripts\python -m chess_agent.match --agent alpha --opponent random --games 20 --depth 3
```

Stockfish 약화 버전과 비교하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.match --agent alpha --opponent uci --opponent-engine stockfish --games 20 --depth 3 --engine-time 0.1 --opponent-engine-option "UCI_LimitStrength=true" --opponent-engine-option "UCI_Elo=1320"
```

색을 번갈아 두지 않고 고정하려면 `--fixed-colors`를 붙입니다.

## 테스트 실행

```powershell
.\.venv\Scripts\python -m pytest
```
