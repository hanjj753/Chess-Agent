# RL 실험 메모

이 폴더는 체스 에이전트를 강화학습으로 학습시키기 위한 첫 실험 공간입니다.
현재 목표는 full chess self-play가 아니라, 작은 `mate-in-1` 퍼즐 환경에서 다음 흐름을 이해하는 것입니다.

```text
Gymnasium 환경
-> observation / action_mask 생성
-> PyTorch policy가 action logits 출력
-> 불법 수를 mask로 제거
-> action 선택
-> reward로 policy 업데이트
```

## 현재 구성

- `mate_in_one_env.py`: Gymnasium 형식의 한 수짜리 체크메이트 퍼즐 환경
- `actions.py`: `chess.Move`와 정수 action id 사이의 변환
- `observations.py`: `chess.Board`를 `18 x 8 x 8` 텐서로 변환
- `random_baseline.py`: 합법 수 중 랜덤 선택 기준선
- `policy.py`: 작은 PyTorch policy network
- `train_mate_in_one.py`: REINFORCE 방식의 간단한 policy-gradient 학습 루프
- `evaluate_mate_in_one.py`: random/policy 평가용 CLI

## 설치

프로젝트 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

현재 필요한 주요 패키지는 다음과 같습니다.

- `python-chess`
- `gymnasium`
- `torch`
- `pytest`

## 랜덤 기준선 평가

먼저 학습하지 않은 랜덤 agent가 어느 정도 맞히는지 봅니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_mate_in_one --agent random --episodes 100 --seed 0
```

출력 예시는 다음 형태입니다.

```text
Mate-in-one evaluation
Agent:          random
Episodes:       100
Successes:      9
Success rate:   9.0%
Illegal moves:  0
Average reward: -0.820
```

여기서 `Illegal moves`가 0이면 action mask가 잘 작동하고 있다는 뜻입니다.

## Policy 학습

현재 학습 코드는 아주 단순한 REINFORCE 방식입니다.
한 episode는 퍼즐 하나이고, 한 수를 두면 바로 끝납니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one --episodes 1000 --hidden-size 64 --learning-rate 0.003 --log-every 250 --seed 0
```

출력 예시는 다음 형태입니다.

```text
episode=  250 train_success=39.2% eval_success=100.0% avg_reward=-0.216
episode=  500 train_success=67.0% eval_success=100.0% avg_reward=0.340
episode=  750 train_success=78.0% eval_success=100.0% avg_reward=0.560
episode= 1000 train_success=83.5% eval_success=100.0% avg_reward=0.670

Training summary
Episodes:             1000
Training success:     83.5%
Final eval success:   100.0%
Final average reward: 1.000
```

현재 기본 퍼즐이 4개뿐이라 `100%`는 "체스를 잘 둔다"가 아니라 "작은 환경에서 학습 루프가 제대로 돈다"는 신호로 해석해야 합니다.

## 모델 저장과 평가

학습한 policy를 저장하려면 `--save-path`를 붙입니다.
`tmp/`는 gitignore되어 있으므로 임시 실험 결과를 저장하기 좋습니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one --episodes 1000 --hidden-size 64 --learning-rate 0.003 --save-path tmp\mate_in_one_policy.pt
```

저장한 모델을 다시 평가하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_mate_in_one --agent policy --model-path tmp\mate_in_one_policy.pt --episodes 4
```

GPU에 저장 모델을 올려서 평가하려면 `--device cuda`를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_mate_in_one --agent policy --model-path tmp\mate_in_one_policy.pt --episodes 4 --device cuda
```

## GPU/CUDA 사용

현재 코드에는 이미 device 옵션이 있습니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one --episodes 1000 --device cuda
```

다만 이 명령이 작동하려면 PyTorch가 CUDA 지원 빌드로 설치되어 있어야 합니다.
현재 설치 상태는 다음 명령으로 확인합니다.

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

예를 들어 다음처럼 나오면 CPU 빌드입니다.

```text
2.13.0+cpu
False
None
```

CUDA를 쓰려면 NVIDIA GPU와 호환 드라이버가 필요하고, PyTorch를 CUDA wheel로 다시 설치해야 합니다.
정확한 설치 명령은 PyTorch 공식 설치 페이지에서 Windows, Pip, Python, CUDA 버전을 선택해서 확인합니다.

https://docs.pytorch.org/get-started/locally/

프로젝트에는 예시용 CUDA requirements 파일도 하나 추가해두었습니다.
이 파일은 CUDA 12.6 wheel index를 사용합니다.

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch
.\.venv\Scripts\python -m pip install -r requirements-torch-cu126.txt
```

하나의 `requirements.txt`가 자동으로 CUDA 버전을 감지해서 맞는 PyTorch를 설치하게 하기는 어렵습니다.
PyTorch는 CUDA 런타임별로 wheel index URL이 달라질 수 있으므로, 보통 CPU용 기본 requirements와 CUDA 버전별 보조 requirements를 분리합니다.

설치 후 `torch.cuda.is_available()`이 `True`가 되면 `--device cuda`를 사용할 수 있습니다.

주의할 점:

- GPU는 policy network의 forward/backward를 빠르게 합니다.
- `python-chess`의 합법 수 계산과 `env.step()`은 CPU에서 돌아갑니다.
- 지금처럼 퍼즐 4개짜리 작은 실험에서는 GPU 이득이 거의 없습니다.
- 퍼즐이 많아지고 batch 학습을 하거나 모델이 커지면 GPU 이득이 커집니다.

## 현재 기본 퍼즐 개수

맞습니다. 현재 기본 퍼즐은 4개뿐입니다.
위치는 `mate_in_one_env.py`의 `DEFAULT_MATE_IN_ONE_FENS`입니다.

현재 퍼즐:

```text
7k/8/5KQ1/8/8/8/8/8 w - - 0 1
8/8/8/8/8/5kq1/8/7K b - - 0 1
6k1/8/6K1/8/8/8/8/R7 w - - 0 1
r7/8/8/8/8/6k1/8/6K1 b - - 0 1
```

다음 단계에서는 이 퍼즐들을 코드에 직접 박아두는 대신, `data/` 아래의 텍스트나 CSV 파일에서 mate-in-1 FEN 목록을 읽어오게 만드는 것이 좋습니다.

## mate-in-1 퍼즐을 구하는 방법

가장 추천하는 출처는 Lichess puzzle database입니다.
Lichess는 퍼즐 CSV를 공개하고 있고, 각 퍼즐에는 `FEN`, `Moves`, `Themes`가 들어 있습니다.

https://database.lichess.org/#puzzles

이 데이터셋에서 `Themes`에 `mateIn1`이 들어간 행만 필터링하면 mate-in-1 퍼즐을 만들 수 있습니다.
주의할 점은 Lichess puzzle CSV의 `FEN`은 플레이어가 바로 풀 위치가 아니라, 상대가 먼저 한 수를 두기 전의 위치입니다.
따라서 CSV의 `Moves` 중 첫 번째 UCI move를 `FEN`에 적용한 뒤, 그 다음 위치를 우리 `ChessMateInOneEnv`에 넣어야 합니다.

예시 흐름:

```text
CSV 한 줄 읽기
-> Themes에 mateIn1이 있는지 확인
-> board = chess.Board(FEN)
-> Moves의 첫 번째 move를 board에 push
-> 이 board.fen()을 mate-in-1 puzzle FEN으로 저장
```

큰 퍼즐 셋을 쓰려면 Lichess CSV에서 mate-in-1 FEN만 추출해서 `data/mate_in_one_fens.txt`로 저장하면 됩니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_mate_in_one_puzzles data\puzzles\lichess_db_puzzle.csv.zst --output data\mate_in_one_fens.txt --limit 10000
```

난이도 범위를 제한하고 싶으면 rating 필터를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_mate_in_one_puzzles data\puzzles\lichess_db_puzzle.csv.zst --output data\mate_in_one_fens.txt --limit 10000 --min-rating 800 --max-rating 1800
```

정답 move와 rating도 같이 저장하고 싶으면 `--include-solution`을 붙입니다.
이 경우 각 줄은 `FEN<TAB>solution_uci<TAB>rating` 형태가 됩니다.
환경은 첫 번째 탭 앞의 FEN만 읽으므로 그대로 사용할 수 있습니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_mate_in_one_puzzles data\puzzles\lichess_db_puzzle.csv.zst --output data\mate_in_one_fens.txt --limit 10000 --include-solution
```

`.csv`, `.csv.gz`, `.csv.zst` 입력을 지원합니다.
`.csv.zst`를 읽으려면 `zstandard` 패키지가 필요하며, `requirements.txt`에 포함되어 있습니다.

추출한 퍼즐 파일로 random baseline을 평가하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_mate_in_one --agent random --puzzles-file data\mate_in_one_fens.txt --episodes 1000
```

추출한 퍼즐 파일로 학습하려면:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one --puzzles-file data\mate_in_one_fens.txt --episodes 10000 --hidden-size 256 --learning-rate 0.001 --log-every 500
```

## 테스트

RL 관련 테스트만 돌리려면:

```powershell
.\.venv\Scripts\python -m pytest tests/test_rl_actions.py tests/test_rl_mate_in_one_env.py tests/test_rl_random_baseline.py tests/test_rl_policy.py
```

전체 테스트:

```powershell
.\.venv\Scripts\python -m pytest
```

## 읽을 때 중요한 코드

처음에는 아래 순서로 읽는 것이 좋습니다.

1. `mate_in_one_env.py`
2. `actions.py`
3. `policy.py`
4. `train_mate_in_one.py`

특히 `train_mate_in_one.py`의 아래 한 줄이 policy-gradient의 핵심입니다.

```python
loss = -distribution.log_prob(action) * reward_tensor
```

선택한 action의 reward가 좋으면 그 action의 확률을 올리고, reward가 나쁘면 그 action의 확률을 내리는 방식입니다.
