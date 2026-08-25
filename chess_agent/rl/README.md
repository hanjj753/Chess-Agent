# RL 실험 메모

이 폴더는 체스 에이전트를 강화학습으로 학습시키기 위한 실험 공간입니다.
현재 목표는 full chess self-play가 아니라, 작은 `mate-in-1` 퍼즐 환경에서 다음 흐름을 익히는 것입니다.

```text
Gymnasium 환경
-> observation / action_mask 생성
-> PyTorch policy가 action logits 출력
-> 불법 수를 mask로 제거
-> action 선택
-> reward 또는 label로 policy 업데이트
```

중요한 원칙은 학습용 퍼즐과 평가용 퍼즐을 처음부터 분리하는 것입니다.
`validation` 파일은 모델 선택과 성능 확인에만 쓰고, 학습에는 넣지 않습니다.

## 현재 구성

- `mate_in_one_env.py`: Gymnasium 형식의 한 수짜리 체크메이트 퍼즐 환경
- `actions.py`: `chess.Move`와 정수 action id 사이의 변환
- `observations.py`: `chess.Board`를 `18 x 8 x 8` 텐서로 변환
- `random_baseline.py`: 합법 수 중 랜덤 선택 기준선
- `policy.py`: 작은 PyTorch policy network
- `train_mate_in_one_supervised.py`: 정답 수를 label로 쓰는 supervised 사전훈련
- `train_mate_in_one.py`: REINFORCE 방식의 간단한 policy-gradient 학습 루프
- `evaluate_mate_in_one.py`: random/policy 평가용 CLI
- `tactical_puzzle_env.py`: 여러 수짜리 Lichess tactical line 환경
- `train_tactical_supervised.py`: 여러 수짜리 tactical puzzle supervised 학습
- `evaluate_tactical.py`: tactical puzzle 평가용 CLI

## 설치

프로젝트 루트에서 실행합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

가상환경을 활성화한 뒤에는 두 운영체제 모두 `python -m ...` 형식으로 실행할 수도 있습니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

현재 필요한 주요 패키지는 다음과 같습니다.

- `python-chess`
- `gymnasium`
- `torch`
- `zstandard`
- `pytest`

## Train/Validation 퍼즐 만들기

추천 데이터 출처는 Lichess puzzle database입니다.

https://database.lichess.org/#puzzles

Lichess puzzle CSV의 `FEN`은 플레이어가 바로 풀 위치가 아니라, 상대가 먼저 한 수를 두기 전의 위치입니다.
그래서 추출 스크립트는 `Moves`의 첫 번째 수를 먼저 적용하고, 그 다음 위치와 정답 수를 저장합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_mate_in_one_puzzles data\puzzles\lichess_db_puzzle.csv.zst --train-output data\puzzle_processed\mate_in_one_train.txt --validation-output data\puzzle_processed\mate_in_one_valid.txt --validation-fraction 0.1 --limit 100000 --include-solution --seed 0
```

각 줄은 `--include-solution`을 붙이면 아래 형식으로 저장됩니다.

```text
FEN<TAB>solution_uci<TAB>rating
```

`ChessMateInOneEnv`는 첫 번째 탭 앞의 FEN만 읽기 때문에, 같은 파일을 환경에도 그대로 넣을 수 있습니다.

처음에는 `--limit 100000` 정도로 실험하고, 서버에서 오래 돌릴 때 `200000`, `500000`처럼 늘리는 편이 좋습니다.
전체를 다 쓰고 싶으면 `--limit`을 빼면 되지만, 파일 생성과 학습 시간이 꽤 길어질 수 있습니다.
split은 streaming random 방식이라 아주 작은 `--limit`에서는 비율이 정확히 맞지 않을 수 있지만, 데이터가 커지면 `--validation-fraction`에 가까워집니다.

난이도 범위를 제한하고 싶으면 rating 필터를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_mate_in_one_puzzles data\puzzles\lichess_db_puzzle.csv.zst --train-output data\puzzle_processed\mate_in_one_train_800_1800.txt --validation-output data\puzzle_processed\mate_in_one_valid_800_1800.txt --validation-fraction 0.1 --limit 100000 --min-rating 800 --max-rating 1800 --include-solution --seed 0
```

## Tactical Puzzle 만들기

mate-in-1 다음 단계는 Lichess puzzle의 여러 수짜리 tactical line을 쓰는 것입니다.
이 데이터는 한 position에서 한 수만 맞히는 것이 아니라, agent move와 opponent reply가 번갈아 나오는 sequence입니다.

저장 형식은 다음과 같습니다.

```text
initial_fen<TAB>line_uci<TAB>rating<TAB>themes
```

여기서 `line_uci`의 짝수 번째 move는 agent가 둘 정답 수이고, 홀수 번째 move는 환경이 자동으로 두는 opponent reply입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_tactical_puzzles data\puzzles\lichess_db_puzzle.csv.zst --train-output data\puzzle_processed\tactical_train.txt --validation-output data\puzzle_processed\tactical_valid.txt --validation-fraction 0.1 --limit 200000 --min-agent-moves 2 --max-agent-moves 4 --min-rating 800 --max-rating 2200 --seed 0
```

기본 theme는 `mateIn2`, `mateIn3`, `fork`, `pin`, `skewer`, `sacrifice`, `discoveredAttack`, `deflection`, `attraction`, `clearance`, `intermezzo`, `trappedPiece`, `xRayAttack`입니다.
다른 theme를 직접 지정하려면 `--themes`를 붙입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.utils.extract_tactical_puzzles data\puzzles\lichess_db_puzzle.csv.zst --train-output data\puzzle_processed\tactical_train_mate.txt --validation-output data\puzzle_processed\tactical_valid_mate.txt --themes mateIn2 mateIn3 mateIn4 --min-agent-moves 2 --limit 100000 --seed 0
```

## 랜덤 기준선 평가

학습 전에는 랜덤 agent를 평가해서 환경과 action mask가 정상인지 확인합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_mate_in_one --agent random --puzzles-file data\puzzle_processed\mate_in_one_valid.txt --episodes 10000 --seed 0
```

`Illegal moves`가 0이면 action mask가 잘 작동하고 있다는 뜻입니다.

## Supervised 사전훈련

mate-in-1 퍼즐은 정답 수가 있는 문제이므로, reward만 보고 맞히기를 기다리는 RL보다 supervised learning으로 먼저 policy를 훈련시키는 편이 훨씬 빠릅니다.
여기서 supervised learning은 최종 목표가 아니라, RL이 랜덤 policy에서 시작하지 않도록 초기 policy를 만들어주는 역할입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one_supervised --puzzles-file data\puzzle_processed\mate_in_one_train.txt --validation-file data\puzzle_processed\mate_in_one_valid.txt --epochs 20 --batch-size 512 --hidden-size 512 --learning-rate 0.001 --device cuda --save-path tmp\mate_in_one_supervised.pt --checkpoint-path tmp\mate_in_one_supervised_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp\mate_in_one_supervised_best.pt
```

저장한 supervised policy를 validation 파일에서 평가합니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_mate_in_one --agent policy --model-path tmp\mate_in_one_supervised.pt --puzzles-file data\puzzle_processed\mate_in_one_valid.txt --episodes 10000 --device cuda
```

## RL Fine-Tuning

supervised로 만든 policy를 초기값으로 넣고 RL을 이어서 돌릴 수 있습니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one --puzzles-file data\puzzle_processed\mate_in_one_train.txt --evaluation-puzzles-file data\puzzle_processed\mate_in_one_valid.txt --evaluation-episodes 5000 --pretrained-path tmp\mate_in_one_supervised.pt --episodes 100000 --hidden-size 512 --learning-rate 0.00003 --log-every 1000 --device cuda --save-path tmp\mate_in_one_rl.pt --checkpoint-path tmp\mate_in_one_rl_checkpoint.pt --checkpoint-every 5000
```

`--evaluation-puzzles-file`은 학습 중 로그와 최종 요약에 사용할 평가 파일입니다.
`--evaluation-episodes`를 지정하면 validation 전체가 아니라 그 개수만큼만 순회합니다.
validation 파일이 커질수록 이 옵션을 지정하는 것이 좋습니다.

현재 RL fine-tuning은 아주 단순한 one-step REINFORCE입니다.
supervised로 80% 이상까지 올린 모델을 이 방식으로 오래 돌리면, 샘플링으로 고른 틀린 수의 negative reward가 이미 배운 분포를 망가뜨려 train/eval 성공률이 내려갈 수 있습니다.
이럴 때는 learning rate를 크게 낮추고 짧게만 실험하거나, supervised checkpoint를 기준으로 다시 시작하는 편이 낫습니다.

중간부터 이어서 돌리려면 `--resume-from`을 사용합니다.
`--epochs`와 `--episodes`는 추가로 돌릴 양이 아니라 도달할 총량입니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one_supervised --puzzles-file data\puzzle_processed\mate_in_one_train.txt --validation-file data\puzzle_processed\mate_in_one_valid.txt --epochs 50 --batch-size 512 --hidden-size 512 --learning-rate 0.001 --device cuda --save-path tmp\mate_in_one_supervised.pt --checkpoint-path tmp\mate_in_one_supervised_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp\mate_in_one_supervised_best.pt --resume-from tmp\mate_in_one_supervised_checkpoint.pt
```

## Tactical Supervised 학습

일주일 동안 서버에서 오래 돌릴 후보로는 현재 이 흐름이 가장 좋습니다.
mate-in-1보다 어렵고, 여러 수짜리 sequence를 따라가야 하므로 더 의미 있는 representation을 배울 가능성이 큽니다.
기본 policy는 체스판의 공간 관계를 보존하는 CNN이며, `AdamW`, weight decay, dropout을 사용합니다.
validation accuracy가 `--patience` 동안 개선되지 않으면 최대 epoch에 도달하기 전이라도 자동으로 종료합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_tactical_supervised --puzzles-file data\puzzle_processed\tactical_train.txt --validation-file data\puzzle_processed\tactical_valid.txt --epochs 500 --batch-size 256 --architecture cnn --hidden-size 64 --residual-blocks 3 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 --patience 15 --evaluation-episodes 10000 --device cuda --save-path tmp\tactical_supervised_cnn.pt --checkpoint-path tmp\tactical_supervised_cnn_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp\tactical_supervised_cnn_best.pt
```

Linux:

```bash
python -m chess_agent.rl.train_tactical_supervised --puzzles-file data/puzzle_processed/tactical_train.txt --validation-file data/puzzle_processed/tactical_valid.txt --epochs 500 --batch-size 256 --architecture cnn --hidden-size 64 --residual-blocks 3 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 --patience 15 --evaluation-episodes 10000 --device cuda --save-path tmp/tactical_supervised_cnn.pt --checkpoint-path tmp/tactical_supervised_cnn_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp/tactical_supervised_cnn_best.pt
```

`--hidden-size`는 CNN에서는 fully connected layer 크기가 아니라 channel 수입니다.
GPU 메모리가 충분하면 `--batch-size 512`, 부족하면 `128`로 조절합니다.
조기 종료를 끄려면 `--patience 0`을 사용합니다.

서버 중단 등으로 checkpoint에서 이어서 돌릴 때 `--epochs`는 추가 epoch가 아니라 도달할 총 epoch입니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_tactical_supervised --puzzles-file data\puzzle_processed\tactical_train.txt --validation-file data\puzzle_processed\tactical_valid.txt --epochs 500 --batch-size 256 --device cuda --save-path tmp\tactical_supervised_cnn.pt --checkpoint-path tmp\tactical_supervised_cnn_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp\tactical_supervised_cnn_best.pt --resume-from tmp\tactical_supervised_cnn_checkpoint.pt
```

Linux:

```bash
python -m chess_agent.rl.train_tactical_supervised --puzzles-file data/puzzle_processed/tactical_train.txt --validation-file data/puzzle_processed/tactical_valid.txt --epochs 500 --batch-size 256 --device cuda --save-path tmp/tactical_supervised_cnn.pt --checkpoint-path tmp/tactical_supervised_cnn_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp/tactical_supervised_cnn_best.pt --resume-from tmp/tactical_supervised_cnn_checkpoint.pt
```

학습한 best tactical policy를 평가하려면:

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_tactical --agent policy --model-path tmp\tactical_supervised_cnn_best.pt --puzzles-file data\puzzle_processed\tactical_valid.txt --episodes all --device cuda --output-path analysis\tactical_evaluation_cnn.txt
```

Linux:

```bash
python -m chess_agent.rl.evaluate_tactical --agent policy --model-path tmp/tactical_supervised_cnn_best.pt --puzzles-file data/puzzle_processed/tactical_valid.txt --episodes all --device cuda --output-path analysis/tactical_evaluation_cnn.txt
```

`--episodes all`은 validation 파일의 각 퍼즐을 정확히 한 번씩 평가합니다.
숫자를 지정했는데 validation 퍼즐보다 크면 처음부터 순환하며 일부 퍼즐을 반복 평가하므로, 최종 비교에는 `all`을 권장합니다.
`--output-path`를 지정하면 콘솔과 같은 내용을 UTF-8 TXT로 저장하며 상위 폴더가 없으면 자동으로 만듭니다.

평가 결과에는 전체 성능과 함께 다음 breakdown이 자동으로 출력됩니다.

- `Rating breakdown`: 200점 단위 Lichess rating 구간
- `Agent move-count breakdown`: 퍼즐을 끝내기 위해 맞혀야 하는 agent move 개수
- `Theme breakdown`: `fork`, `pin`, `mateIn2` 같은 전술 주제별 결과
- `Difficulty-adjusted theme breakdown`: rating 구간과 move count를 보정한 theme별 결과

한 퍼즐에는 theme가 여러 개 붙을 수 있으므로 theme별 `Episodes` 합계는 전체 episode 수보다 클 수 있습니다.
기본적으로 episode가 20개보다 적은 theme는 숨깁니다. 모든 theme를 보려면 Windows와 Linux 모두 평가 명령에 `--min-theme-episodes 0`을 추가합니다.
Theme 표는 성공률이 낮은 순서로 출력되므로 취약한 전술을 위에서부터 확인할 수 있습니다.

난이도 보정 표의 `Expected`는 같은 `rating 200점 구간 + agent move count`에 속한 전체 퍼즐의 평균으로 계산합니다.
`Success gap`과 `Move gap`은 `실제 - 기대`이며 음수일수록 난이도와 길이를 고려한 뒤에도 해당 theme에 약하다는 뜻입니다.
한 퍼즐에 theme가 여러 개 붙을 수 있으므로 이 결과는 theme의 인과 효과가 아니라 취약 영역을 찾기 위한 진단 지표로 해석합니다.

`Validation accuracy`는 각 agent turn에서 정답 수를 맞힌 비율이고, `Validation puzzle success`는 한 퍼즐의 sequence를 끝까지 모두 맞힌 비율입니다.
여러 수를 연속으로 맞혀야 하므로 puzzle success는 move accuracy보다 훨씬 낮게 나오는 것이 자연스럽습니다.
학습 로그에는 `best_val_acc`와 `best_epoch`가 계속 출력됩니다.
`--save-path`는 마지막 epoch 모델이고, `--best-checkpoint-path`는 validation accuracy가 가장 좋았던 epoch의 모델입니다.
기존 MLP checkpoint는 계속 불러오고 평가할 수 있지만 CNN으로 변환되지는 않습니다. CNN 학습은 위 명령처럼 새 파일명으로 시작합니다.

## GPU/CUDA 사용

현재 코드에는 이미 device 옵션이 있습니다.

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_mate_in_one_supervised --puzzles-file data\puzzle_processed\mate_in_one_train.txt --validation-file data\puzzle_processed\mate_in_one_valid.txt --device cuda
```

설치 상태는 다음 명령으로 확인합니다.

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

`torch.cuda.is_available()`이 `True`이면 `--device cuda`를 사용할 수 있습니다.
CUDA 빌드 PyTorch가 필요하면 아래 보조 requirements 중 서버 환경에 맞는 것을 설치합니다.

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch
.\.venv\Scripts\python -m pip install -r requirements-torch-cu130.txt
```

서버의 드라이버가 CUDA 12.6 쪽에 맞으면 다음 파일을 사용할 수도 있습니다.

```powershell
.\.venv\Scripts\python -m pip install -r requirements-torch-cu126.txt
```

주의할 점:

- GPU는 PyTorch network의 forward/backward를 빠르게 합니다.
- `python-chess`의 합법 수 계산과 environment logic은 CPU에서 돌아갑니다.
- 작은 퍼즐 4개짜리 smoke test에서는 GPU 이득이 거의 없습니다.
- 퍼즐이 많아지고 batch supervised 학습을 할수록 GPU 이득이 커집니다.

## 오래 학습시킬 때 추천 흐름

1. 먼저 `--limit 100000`으로 추출해서 전체 파이프라인을 확인합니다.
2. supervised 사전훈련을 `20~50` epochs 정도 돌립니다.
3. validation accuracy와 mate success가 충분히 오르는지 확인합니다.
4. 서버에서는 `--limit 200000~500000`으로 데이터를 늘립니다. MLP는 `--hidden-size 512`, CNN은 `--hidden-size 64~128` 정도가 출발점입니다.
5. checkpoint를 켠 상태로 supervised 학습을 길게 돌립니다.
6. RL fine-tuning은 작은 learning rate로 짧게 붙이고, 성능이 내려가면 멈춥니다.

일주일 동안 서버에 걸어둘 작업으로는 현재 코드 기준에서 `mate-in-1` supervised 대규모 학습이 가장 안정적입니다.
다만 이 작업만으로 full chess agent가 되지는 않습니다.
이제 다음 장기 실험 후보로 Lichess puzzle의 `mateIn2`, `mateIn3`, `fork`, `pin`, `skewer`, `sacrifice` 같은 tactical theme를 다루는 multi-step puzzle 환경도 사용할 수 있습니다.

## 기본 퍼즐

데이터 파일을 넣지 않으면 `mate_in_one_env.py`의 `DEFAULT_MATE_IN_ONE_FENS` 4개만 사용합니다.
이 4개는 학습 성능 측정용이 아니라, 코드가 돌아가는지 확인하는 smoke test용입니다.

## 테스트

RL 관련 테스트만 돌리려면:

```powershell
.\.venv\Scripts\python -m pytest tests/test_rl_actions.py tests/test_rl_mate_in_one_env.py tests/test_rl_random_baseline.py tests/test_rl_policy.py tests/test_rl_extract_mate_in_one_puzzles.py tests/test_rl_supervised_training.py
```

tactical puzzle 관련 테스트만 돌리려면:

```powershell
.\.venv\Scripts\python -m pytest tests/test_rl_tactical_puzzle_env.py tests/test_rl_extract_tactical_puzzles.py tests/test_rl_tactical_supervised_training.py
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
4. `train_mate_in_one_supervised.py`
5. `train_mate_in_one.py`
6. `tactical_puzzle_env.py`
7. `train_tactical_supervised.py`

`train_mate_in_one_supervised.py`에서는 정답 move를 cross entropy loss로 맞히고, `train_mate_in_one.py`에서는 아래 한 줄이 policy-gradient의 핵심입니다.

```python
loss = -distribution.log_prob(action) * reward_tensor
```

선택한 action의 reward가 좋으면 그 action의 확률을 올리고, reward가 나쁘면 그 action의 확률을 내리는 방식입니다.
