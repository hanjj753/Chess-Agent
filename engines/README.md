# 외부 체스 엔진 사용법

이 폴더에는 다운로드한 외부 UCI 엔진 실행 파일을 둡니다.
엔진 바이너리와 네트워크 가중치 파일은 용량이 크고 플랫폼별로 달라서 Git에는 올리지 않습니다.

## Stockfish

- UCI 표준을 지원해서 `python-chess`와 바로 연결됩니다.
- 매우 강해서 기준 엔진으로 쓰기 좋습니다.
- `UCI_Elo`와 `Skill Level`로 약하게 만들 수 있어 단계별 평가가 가능합니다.
- 강화학습 전에도 우리 에이전트의 실력을 꾸준히 비교할 수 있습니다.

공식 다운로드:

- https://stockfishchess.org/download/
- https://github.com/official-stockfish/Stockfish/releases/latest

Windows 예시 폴더:

```text
engines/
  stockfish/
    stockfish-windows-x86-64-avx2.exe
```

기본 실행:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine stockfish --depth 3 --engine-time 0.1
```

Stockfish를 Elo 기준으로 약하게 설정:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine stockfish --depth 3 --engine-time 0.1 --black-engine-option "UCI_LimitStrength=true" --black-engine-option "UCI_Elo=1320"
```

추천 벤치마크 단계:

```text
1320 -> 1600 -> 2000 -> 2400 -> 2800
```

더 단순하게는 `Skill Level`을 쓸 수 있습니다.

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine stockfish --depth 3 --engine-time 0.1 --black-engine-option "Skill Level=3"
```

참고:

- `UCI_LimitStrength=true`와 `UCI_Elo`를 쓰면 `UCI_Elo`가 `Skill Level`보다 우선합니다.
- `Skill Level`은 `0`부터 `20`까지이고, 낮을수록 약하게 둡니다.
- 같은 설정이라도 짧은 시간 제한에서는 결과가 흔들릴 수 있으니 여러 판 평균을 봐야 합니다.

## Lc0

Lc0는 신경망 기반 UCI 엔진입니다.
강화학습이나 신경망 기반 체스 엔진을 공부할 때 좋은 비교 대상입니다.

공식 다운로드:

- https://lczero.org/play/download/
- https://github.com/LeelaChessZero/lc0/releases

예시 폴더:

```text
engines/
  lc0/
    lc0.exe
    703810.pb.gz
```

실행 예시:

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine lc0 --depth 3 --engine-time 0.1
```

네트워크 파일은 Lc0 패키지에 기본 포함된 것을 먼저 쓰고, 나중에 필요하면 별도 네트워크를 받아 바꾸면 됩니다.

## Maia

Maia는 인간 기보로 학습된 Lc0용 네트워크입니다.
강한 엔진이라기보다 특정 Elo대 인간처럼 두는 모델에 가깝습니다.

공식 자료:

- https://github.com/CSSLab/maia-chess
- https://lczero.org/play/networks/sparring-nets/

Maia는 단독 실행 파일이 아니라 Lc0에 얹는 네트워크 파일입니다.

예시 폴더:

```text
engines/
  lc0-maia-1500/
    lc0.exe
    maia-1500.pb.gz
```

Maia는 보통 `nodes=1`로 실행합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.play --white alpha --black uci --black-engine lc0-maia-1500 --engine-nodes 1
```

Maia는 객관적인 최강 기준보다는, “사람 같은 상대와 두었을 때 얼마나 잘 버티는지”를 보는 용도에 더 어울립니다.
