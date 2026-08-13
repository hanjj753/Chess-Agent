# 체스 에이전트 학습 프로젝트

이 프로젝트는 체스를 두는 에이전트를 만들면서, 완성된 코드를 그냥 받아쓰기보다
핵심 아이디어를 직접 이해하고 바꿔보는 것을 목표로 합니다.

## 현재 기준 버전

- `python-chess`로 체스 규칙 처리
- 랜덤 에이전트
- 사람이 직접 수를 입력하는 에이전트
- 외부 UCI 엔진 실행용 에이전트
- negamax + alpha-beta pruning + quiescence search + transposition table 기반 에이전트
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

만약 GPU 를 사용하는 환경이라면 아래 명령어를 쳐 CUDA 버전을 확인한 뒤 알맞는 버전의 requirements-torch.txt를 설치하면 됩니다.
다시 말해,

```powershell
nvidia-smi
```
위 명령어로 CUDA 버전을 확인한 후, CUDA 13.0 아래면

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch
.\.venv\Scripts\python -m pip install -r requirements-torch-cu126.txt
```

CUDA 13.0 이상이면
```powershell
.\.venv\Scripts\python -m pip uninstall -y torch
.\.venv\Scripts\python -m pip install -r requirements-torch-cu130.txt
```


## 대국 실행

Alpha-beta 에이전트와 랜덤 에이전트를 대국시키려면:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black random --depth 3
```

시간 기준으로 탐색하려면 `--time-limit`을 붙입니다.
이 경우 depth 1부터 `--depth`까지 차례로 깊게 보다가 제한 시간이 되면 마지막으로 완성한 깊이의 수를 둡니다.

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black random --depth 8 --time-limit 0.5
```

사람이 백으로 직접 두고 alpha-beta 에이전트와 대국하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white human --black alpha --depth 3
```

사람이 포함된 대국은 기본적으로 GUI 보드가 열립니다.
터미널 입력으로 두고 싶으면 `--no-gui`를 붙입니다.

입력할 수의 예시는 다음과 같습니다.

- SAN 표기: `e4`, `Nf3`, `O-O`, `Qxe5+`
- UCI 표기: `e2e4`, `g1f3`, `e7e8q`
- 종료: `quit`, `exit`, `resign`

외부 UCI 엔진과 대국하려면 엔진 실행 파일을 `engines/` 아래에 넣고 `uci` 에이전트를 사용합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine stockfish --depth 3 --engine-time 0.1
```

자세한 폴더 예시는 [engines/README.md](<D:/Workspace/Chess/engines/README.md>)를 참고하면 됩니다.

에이전트끼리 두는 대국을 GUI로 보고 싶으면 `--gui`를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black random --depth 3 --gui --gui-delay-ms 500
```

## 여러 판 평가 실행

한 판이 아니라 여러 판을 돌려 승/무/패와 점수율을 보려면 `match` 모드를 사용합니다.
기본적으로 매 게임마다 백/흑을 번갈아 둡니다.
각 게임마다 우리 에이전트가 탐색한 `nodes`와 transposition table 재사용 횟수인 `tt_hits`도 출력합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.match --agent alpha --opponent random --games 6 --depth 3
```

Stockfish 약화 버전과 비교하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.match --agent alpha --opponent uci --opponent-engine stockfish --games 10 --depth 3 --engine-time 0.1 --opponent-engine-option "UCI_LimitStrength=true" --opponent-engine-option "UCI_Elo=2000"
```

색을 번갈아 두지 않고 고정하려면 `--fixed-colors`를 붙입니다.

진 경기만 PGN으로 저장하려면 `--save-losses`를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.match --agent alpha --opponent uci --opponent-engine stockfish --games 10 --depth 8 --time-limit 0.5 --engine-time 0.1 --opponent-engine-option "UCI_LimitStrength=true" --opponent-engine-option "UCI_Elo=1600" --save-losses analysis/losses
```

## 진 경기 분석

저장된 PGN을 Stockfish로 분석하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.analyze_game analysis/losses/loss_파일명.pgn --engine stockfish --engine-time 0.1
```

분석 결과를 GUI로 넘겨보려면 `--gui`를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.analyze_game analysis/losses/loss_파일명.pgn --engine stockfish --engine-time 0.1 --gui
```

폴더 안의 패배 PGN들을 한 번에 요약하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.analyze_batch analysis/losses --engine stockfish --engine-time 0.1
```

이미 `.analysis.json` 파일이 있으면 기본적으로 재사용합니다.
다시 분석하려면 `--no-reuse`를 붙입니다.
기본 요약은 파일명에 저장된 색을 기준으로 우리 에이전트가 둔 수만 봅니다.
상대 수까지 같이 보고 싶으면 `--all-moves`를 붙입니다.
mate 평가처럼 100000cp에 가까운 값은 평균을 크게 왜곡하므로, 요약의 `Avg loss`는 기본적으로 1000cp cap을 적용합니다.

## 테스트 실행

```powershell
.\.venv\Scripts\python -m pytest
```
