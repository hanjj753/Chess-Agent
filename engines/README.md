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

### Linux에서 실행할 때

Windows용 `.exe`는 Linux에서 직접 실행할 수 없습니다. 실행 권한만 붙이는
`chmod +x`로 해결되지 않으며 Linux용 바이너리가 필요합니다.

Ubuntu에서는 패키지를 설치하고 실행 파일 경로를 직접 지정할 수 있습니다.

```bash
sudo apt-get update
sudo apt-get install -y stockfish
printf 'uci\nquit\n' | /usr/games/stockfish
```

마지막 명령이 Stockfish 이름과 `uciok`를 출력하면 UCI 실행이 되는 것입니다.
교사 데이터 수집에는 `--engine /usr/games/stockfish`를 사용합니다. `play`에서는
`--white-engine` 또는 `--black-engine`에 같은 절대 경로를 전달합니다.

sudo 권한이 없다면 [공식 다운로드](https://stockfishchess.org/download/)에서
서버 OS/CPU에 맞는 Linux 버전을 받아 압축을 풀고, 실제 바이너리의 절대 경로를
사용합니다. **신뢰할 수 있는 Linux 바이너리**에 실행 권한이 빠졌을 때만
`chmod +x /absolute/path/to/stockfish`를 실행합니다.

폴더를 지정하면 Windows에서는 Windows 실행 파일만, Linux/macOS에서는 Windows
확장자가 아니며 실행 권한이 있는 파일만 고릅니다. 후보가 여러 개라면 임의로 선택하지
않고 실제 실행 파일 경로를 요구합니다. 문서나 가중치 파일에 실행 권한을 부여하지 마세요.

실패한 교사 수집 실행에서 이미 `manifest.json`이 만들어졌다면 엔진 변경 후 기존
폴더로 `--resume`할 수 없습니다. `--output-dir data/teacher/p50_v1_linux`처럼 새
폴더로 시작하면 기존 파일을 삭제할 필요가 없습니다.

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
