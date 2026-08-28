# RL 실험 메모

이 폴더는 체스 에이전트를 강화학습으로 학습시키기 위한 실험 공간입니다.
작은 `mate-in-1`과 여러 수짜리 tactical puzzle로 기본 policy를 사전학습한 뒤,
현재는 full-chess actor-critic 학습을 위한 환경과 Policy-Value CNN으로 확장하고 있습니다.

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
- `full_chess_env.py`: 최근 보드 history와 고정 상대를 사용하는 일반 대국 환경
- `policy_value.py`: 공유 CNN 위에 policy head와 value head를 둔 모델
- `initialize_policy_value.py`: tactical CNN checkpoint를 Policy-Value 모델로 변환
- `value_dataset.py`: full-chess value dataset의 bit-pack NPZ 저장/불러오기
- `collect_value_dataset.py`: mixed 상대 대국에서 value train/validation 상태 수집
- `pretrain_value_head.py`: policy를 고정한 value head supervised 사전학습
- `evaluate_value_head.py`: checkpoint를 지정한 value dataset에서 독립 평가
- `experiment_tracking.py`: 설정, 학습 지표, 대국 결과와 checkpoint 이벤트 기록
- `ppo_policy.py`: 기존 CNN 구조를 사용하는 Stable-Baselines3 maskable policy
- `train_full_chess_ppo.py`: FullChess Maskable PPO 학습, 평가, checkpoint 기록
- `evaluate_full_chess_ppo.py`: 저장된 PPO 모델의 독립 대국 평가와 TXT 보고서
- `compare_full_chess_evaluations.py`: 같은 seed의 두 평가 CSV를 paired 비교
- `report_experiment.py`: 실험 로그를 한국어 TXT와 발표용 PNG로 자동 요약

## Full-Chess 환경

`FullChessEnv`의 한 `step()`은 agent의 수를 적용한 뒤 상대의 응수까지 진행합니다.
따라서 step이 정상적으로 끝났다면 다음 observation도 다시 agent 차례입니다.
상대는 기본 random agent이며 기존 `Agent` 구현을 전달해 교체할 수 있습니다.

기본 `history_length=4`에서는 현재 보드 1개와 직전 보드 4개를 사용합니다.
각 보드는 기존과 같은 18개 plane이므로 입력 shape은 다음과 같습니다.

```text
(18 * (4 + 1), 8, 8) = (90, 8, 8)
```

채널 순서는 `현재 위치 -> 한 ply 전 -> 두 ply 전 -> ...`이며, 게임 초반에
history가 부족한 부분은 0으로 채웁니다. 환경은 체크메이트와 python-chess의
무승부 판정을 terminal로 처리하고, `max_plies`에 도달하면 truncate합니다.
보상은 agent 관점에서 승리 `+1`, 무승부 `0`, 패배 `-1`입니다.

현재 PPO 학습은 고정된 random 또는 tree agent를 상대로 진행합니다. 두 학습 agent가
서로 갱신되는 self-play는 아직 연결하지 않았습니다.

### 학습 단위

현재 `FullChessEnv`에서는 **episode 1개가 체스 대국 1판**입니다. RL에서 episode는
`reset()`부터 terminal 또는 truncate까지의 한 trajectory를 뜻하며, 이 환경에서는
그 시작과 끝이 대국의 시작과 끝에 정확히 대응합니다.

- `ply`: 백 또는 흑이 둔 수 하나
- `timestep`: agent가 action 하나를 고르고 `env.step()`을 호출한 횟수
- `episode`: 게임이 끝날 때까지 이어진 timestep 묶음, 현재는 대국 1판
- `rollout`: PPO가 한 번 학습하기 전에 여러 환경에서 모으는 timestep 묶음
- `batch`: rollout을 network update에 넣기 위해 나눈 조각
- `n_epochs`: 같은 rollout 데이터를 반복 학습하는 횟수

`step()` 안에서 agent 수 뒤에 상대 응수까지 자동으로 진행하므로, 보통 timestep 1개는
약 2 plies입니다. 게임이 rollout 도중 끝나면 환경은 새 episode를 시작해서 rollout에
필요한 timestep 수를 계속 채웁니다. 따라서 rollout 하나에는 여러 대국이 섞일 수 있습니다.

예를 들어 smoke 설정의 `n_envs=1`, `n_steps=256`, `batch_size=256`, `n_epochs=2`는
다음 순서로 작동합니다.

```text
256 timestep 수집 -> 하나의 256개 batch 생성 -> 같은 batch로 PPO update 2회
```

`total_timesteps=4096`라면 이 rollout을 16번 수집합니다. smoke 결과의 학습 대국은
115판이었으므로 rollout 하나에 평균 약 7판이 들어간 셈입니다. 평가 대국은 별도로
결정론적 action을 사용해 실행하며, 그 결과로 network를 업데이트하지 않습니다.

## Policy-Value 초기화

Policy-Value CNN은 기존 CNN 몸통을 공유하고 두 출력을 냅니다.

```text
shared CNN -> policy head: 20,480 action logits
           -> value head:  -1~1 사이의 현재 위치 가치
```

기존 tactical CNN의 `input block`, residual backbone, policy head weight를 복사합니다.
늘어난 history 입력 채널의 첫 convolution weight는 0으로 초기화하므로, 변환 직후
policy 출력은 기존 tactical CNN과 같습니다. value head는 새로 학습해야 합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.initialize_policy_value --policy-path tmp\tactical_supervised_cnn_best.pt --output-path tmp\full_chess_policy_value.pt --history-length 4
```

Linux:

```bash
python -m chess_agent.rl.initialize_policy_value --policy-path tmp/tactical_supervised_cnn_best.pt --output-path tmp/full_chess_policy_value.pt --history-length 4
```

### Value Head Supervised 사전학습

초기 Policy-Value checkpoint의 value head는 정답으로 학습되지 않은 상태입니다. PPO의
희소한 승패 reward만으로 처음부터 배우기 전에, 완결 대국의 각 agent-turn 상태에 최종
결과를 label로 붙여 value head를 먼저 학습합니다. 마지막 상태는 승/무/패를 `+1/0/-1`로
사용하고, 앞선 상태는 PPO와 같은 `gamma=0.995`만큼 할인합니다.

데이터는 같은 대국의 상태가 train과 validation에 나뉘지 않도록 대국 단위로 분리합니다.
observation은 0/1 bit-pack NPZ로 저장해 일반 float tensor보다 디스크 사용량을 줄입니다.
random 상대는 승리 쪽, alpha 상대는 패배 쪽으로 치우칠 수 있어 기본 수집은 둘을 절반씩
섞습니다. Policy는 확률적으로 수를 골라 더 다양한 상태를 만듭니다.

#### 1. Value dataset 수집

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.collect_value_dataset --model-path tmp\full_chess_policy_value.pt --train-output data\value\full_chess_value_train.npz --validation-output data\value\full_chess_value_valid.npz --games 5000 --validation-fraction 0.1 --opponent mixed --alpha-fraction 0.5 --opponent-depth 1 --max-plies 200 --gamma 0.995 --seed 0 --device cuda --log-every 100
```

Linux:

```bash
python -m chess_agent.rl.collect_value_dataset --model-path tmp/full_chess_policy_value.pt --train-output data/value/full_chess_value_train.npz --validation-output data/value/full_chess_value_valid.npz --games 5000 --validation-fraction 0.1 --opponent mixed --alpha-fraction 0.5 --opponent-depth 1 --max-plies 200 --gamma 0.995 --seed 0 --device cuda --log-every 100
```

처음 파이프라인만 확인할 때는 `--games 500`과 별도 `_smoke.npz` 출력 이름을 사용합니다.
최종 W/D/L에서 한 결과가 지나치게 적으면 `--alpha-fraction`이나 `--opponent-depth`를
조절해 양수, 0, 음수 label을 모두 확보합니다.

#### 2. Value head 학습

입력 CNN과 policy head는 eval mode로 고정하며 weight와 BatchNorm 통계를 바꾸지 않습니다.
value head만 Huber loss로 학습합니다. 긴 대국이 과도한 비중을 갖지 않도록 대국별
가중치를 맞추고, 승/무/패 대국 수도 기본적으로 균형화합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.pretrain_value_head --model-path tmp\full_chess_policy_value.pt --train-data data\value\full_chess_value_train.npz --validation-data data\value\full_chess_value_valid.npz --epochs 50 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.00001 --patience 10 --seed 0 --device cuda --save-path tmp\full_chess_policy_value_value_final.pt --best-model-path tmp\full_chess_policy_value_value_best.pt --experiment-dir analysis\experiments --experiment-name value_head_pretrain
```

Linux:

```bash
python -m chess_agent.rl.pretrain_value_head --model-path tmp/full_chess_policy_value.pt --train-data data/value/full_chess_value_train.npz --validation-data data/value/full_chess_value_valid.npz --epochs 50 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.00001 --patience 10 --seed 0 --device cuda --save-path tmp/full_chess_policy_value_value_final.pt --best-model-path tmp/full_chess_policy_value_value_best.pt --experiment-dir analysis/experiments --experiment-name value_head_pretrain
```

우선 볼 값은 `validation_loss`, `validation_explained_variance`, `prediction_std`입니다.
explained variance가 0보다 높아지고 prediction std가 target std 방향으로 증가해야 value가
상태 차이를 배우고 있다고 볼 수 있습니다. `report_experiment`는 이 실험을 자동으로
구분해 value 전용 `summary.txt`와 `learning_curves.png`를 생성합니다.

#### 3. 사전학습 Value로 PPO 비교

value best checkpoint를 사용하되 나머지는 앞선 rollout 1024 실험과 동일하게 둡니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value_value_best.pt --total-timesteps 16384 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp\full_chess_ppo_valuepretrain_initial.zip --save-path tmp\full_chess_ppo_valuepretrain_final.zip --best-model-path tmp\full_chess_ppo_valuepretrain_best.zip --checkpoint-dir tmp\full_chess_ppo_valuepretrain_checkpoints --experiment-name ppo_valuepretrain_rollout1024
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value_value_best.pt --total-timesteps 16384 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp/full_chess_ppo_valuepretrain_initial.zip --save-path tmp/full_chess_ppo_valuepretrain_final.zip --best-model-path tmp/full_chess_ppo_valuepretrain_best.zip --checkpoint-dir tmp/full_chess_ppo_valuepretrain_checkpoints --experiment-name ppo_valuepretrain_rollout1024
```

이 실험에서는 기존 `ppo_rollout1024`와 비교해 step 0의 explained variance, 초반 value loss,
주기 evaluation 점수, best checkpoint가 step 0을 넘어서는지를 확인합니다.

#### 4. 기존 PPO와 500판 paired 비교

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp\full_chess_ppo_rollout1024_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis\ppo_rollout1024_final_500.txt
.\.venv\Scripts\python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp\full_chess_ppo_valuepretrain_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis\ppo_valuepretrain_final_500.txt
.\.venv\Scripts\python -m chess_agent.rl.compare_full_chess_evaluations analysis\ppo_rollout1024_final_500_games.csv analysis\ppo_valuepretrain_final_500_games.csv --output-path analysis\ppo_rollout1024_vs_valuepretrain_final.txt
```

Linux:

```bash
python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp/full_chess_ppo_rollout1024_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis/ppo_rollout1024_final_500.txt
python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp/full_chess_ppo_valuepretrain_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis/ppo_valuepretrain_final_500.txt
python -m chess_agent.rl.compare_full_chess_evaluations analysis/ppo_rollout1024_final_500_games.csv analysis/ppo_valuepretrain_final_500_games.csv --output-path analysis/ppo_rollout1024_vs_valuepretrain_final.txt
```

### PPO 조건 일치 Value 재실험

Value는 상대 정책까지 포함한 기대 reward이므로 random 상대와 alpha 상대에서 서로 다른
함수가 됩니다. PPO가 `random`, `max_plies=100`으로 학습될 때는 value 데이터도 같은
조건으로 만드는 것이 우선입니다. Random 상대에서는 패배가 매우 희소할 수 있으므로
`--no-balance-outcomes`를 사용해 극소수 패배 대국이 과도하게 확대되지 않게 합니다.

#### 1. Random-100 데이터 수집

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.collect_value_dataset --model-path tmp\full_chess_policy_value.pt --train-output data\value\full_chess_value_random100_train.npz --validation-output data\value\full_chess_value_random100_valid.npz --games 5000 --validation-fraction 0.1 --opponent random --max-plies 100 --gamma 0.995 --seed 0 --device cuda --log-every 100
```

Linux:

```bash
python -m chess_agent.rl.collect_value_dataset --model-path tmp/full_chess_policy_value.pt --train-output data/value/full_chess_value_random100_train.npz --validation-output data/value/full_chess_value_random100_valid.npz --games 5000 --validation-fraction 0.1 --opponent random --max-plies 100 --gamma 0.995 --seed 0 --device cuda --log-every 100
```

#### 2. Random-100 Value head 사전학습

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.pretrain_value_head --model-path tmp\full_chess_policy_value.pt --train-data data\value\full_chess_value_random100_train.npz --validation-data data\value\full_chess_value_random100_valid.npz --epochs 50 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.00001 --patience 10 --no-balance-outcomes --seed 0 --device cuda --save-path tmp\full_chess_policy_value_random100_final.pt --best-model-path tmp\full_chess_policy_value_random100_best.pt --experiment-dir analysis\experiments --experiment-name value_head_pretrain_random100
```

Linux:

```bash
python -m chess_agent.rl.pretrain_value_head --model-path tmp/full_chess_policy_value.pt --train-data data/value/full_chess_value_random100_train.npz --validation-data data/value/full_chess_value_random100_valid.npz --epochs 50 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.00001 --patience 10 --no-balance-outcomes --seed 0 --device cuda --save-path tmp/full_chess_policy_value_random100_final.pt --best-model-path tmp/full_chess_policy_value_random100_best.pt --experiment-dir analysis/experiments --experiment-name value_head_pretrain_random100
```

#### 3. 같은 Random-100 validation에서 Value 직접 비교

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_value_head --model-path tmp\full_chess_policy_value_value_best.pt --data data\value\full_chess_value_random100_valid.npz --device cuda --output-path analysis\value_mixed200_on_random100.txt
.\.venv\Scripts\python -m chess_agent.rl.evaluate_value_head --model-path tmp\full_chess_policy_value_random100_best.pt --data data\value\full_chess_value_random100_valid.npz --device cuda --output-path analysis\value_random100_on_random100.txt
```

Linux:

```bash
python -m chess_agent.rl.evaluate_value_head --model-path tmp/full_chess_policy_value_value_best.pt --data data/value/full_chess_value_random100_valid.npz --device cuda --output-path analysis/value_mixed200_on_random100.txt
python -m chess_agent.rl.evaluate_value_head --model-path tmp/full_chess_policy_value_random100_best.pt --data data/value/full_chess_value_random100_valid.npz --device cuda --output-path analysis/value_random100_on_random100.txt
```

두 TXT에서 Huber loss와 MAE는 낮을수록, explained variance는 높을수록 좋습니다. 이
비교는 PPO update가 개입하기 전이라 데이터 조건 일치의 효과만 보여줍니다.

#### 4. Random-100 Value로 PPO 학습

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value_random100_best.pt --total-timesteps 16384 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp\full_chess_ppo_value_random100_initial.zip --save-path tmp\full_chess_ppo_value_random100_final.zip --best-model-path tmp\full_chess_ppo_value_random100_best.zip --checkpoint-dir tmp\full_chess_ppo_value_random100_checkpoints --experiment-name ppo_value_random100_rollout1024
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value_random100_best.pt --total-timesteps 16384 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp/full_chess_ppo_value_random100_initial.zip --save-path tmp/full_chess_ppo_value_random100_final.zip --best-model-path tmp/full_chess_ppo_value_random100_best.zip --checkpoint-dir tmp/full_chess_ppo_value_random100_checkpoints --experiment-name ppo_value_random100_rollout1024
```

#### 5. 기존 PPO와 500판 paired 비교

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp\full_chess_ppo_rollout1024_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis\ppo_rollout1024_final_500.txt
.\.venv\Scripts\python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp\full_chess_ppo_value_random100_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis\ppo_value_random100_final_500.txt
.\.venv\Scripts\python -m chess_agent.rl.compare_full_chess_evaluations analysis\ppo_rollout1024_final_500_games.csv analysis\ppo_value_random100_final_500_games.csv --output-path analysis\ppo_rollout1024_vs_value_random100_final.txt
```

Linux:

```bash
python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp/full_chess_ppo_rollout1024_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis/ppo_rollout1024_final_500.txt
python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp/full_chess_ppo_value_random100_final.zip --games 500 --opponent random --max-plies 100 --seed 10000 --device cuda --output-path analysis/ppo_value_random100_final_500.txt
python -m chess_agent.rl.compare_full_chess_evaluations analysis/ppo_rollout1024_final_500_games.csv analysis/ppo_value_random100_final_500_games.csv --output-path analysis/ppo_rollout1024_vs_value_random100_final.txt
```

## 실험 과정 기록

학습 명령에 `--experiment-dir`을 지정하면 실행마다 timestamp가 붙은 별도 폴더를
만듭니다. `--experiment-name`에는 발표에서 구분하기 쉬운 짧은 실험명을 적습니다.

Windows PowerShell 예시:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_tactical_supervised --puzzles-file data\puzzle_processed\tactical_train.txt --validation-file data\puzzle_processed\tactical_valid.txt --epochs 500 --batch-size 256 --architecture cnn --hidden-size 64 --residual-blocks 3 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 --patience 15 --device cuda --save-path tmp\tactical_supervised_cnn.pt --best-checkpoint-path tmp\tactical_supervised_cnn_best.pt --experiment-dir analysis\experiments --experiment-name tactical_baseline
```

Linux 예시:

```bash
python -m chess_agent.rl.train_tactical_supervised --puzzles-file data/puzzle_processed/tactical_train.txt --validation-file data/puzzle_processed/tactical_valid.txt --epochs 500 --batch-size 256 --architecture cnn --hidden-size 64 --residual-blocks 3 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 --patience 15 --device cuda --save-path tmp/tactical_supervised_cnn.pt --best-checkpoint-path tmp/tactical_supervised_cnn_best.pt --experiment-dir analysis/experiments --experiment-name tactical_baseline
```

생성되는 파일은 다음과 같습니다.

- `config.json`: hyperparameter, 데이터 경로, 모델 설정
- `metrics.csv`: epoch/step별 loss, accuracy, reward, 승률 등의 긴 형식 데이터
- `games.csv`: episode별 승패, 색, reward, ply, 종료 원인
- `events.jsonl`: best checkpoint 저장 같은 시간 순서 이벤트
- `summary.json`: 실험 종료 시점의 최종 요약

현재 tactical trainer는 `config.json`, `metrics.csv`, `events.jsonl`, `summary.json`을
기록합니다. `games.csv`는 파일 형식을 미리 생성하고, 다음 FullChess 학습기에서
각 self-play 결과를 기록할 때 사용합니다. `metrics.csv`가 긴 형식이므로 pandas,
Excel 또는 발표용 그래프 코드에서 `step`, `metric`, `value` 열을 바로 사용할 수 있습니다.

### 자동 실험 보고서

실험 폴더 하나를 지정하면 `report/` 아래에 한국어 요약과 발표에 쓸 그래프를 만듭니다.
상위 `analysis/experiments` 폴더를 지정하면 바로 아래의 모든 실험 폴더에 대해 보고서를
생성합니다.

기본적으로 필요한 `summary.txt`와 그래프가 이미 모두 있는 실험은 건너뜁니다. 일부
보고서 파일이 빠진 실험만 새로 생성하며, 기존 보고서까지 모두 갱신하려면 `--force`를
사용합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.report_experiment analysis\experiments
```

Linux:

```bash
python -m chess_agent.rl.report_experiment analysis/experiments
```

기존 보고서를 포함해 모두 다시 생성하는 명령은 다음과 같습니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.report_experiment analysis\experiments --force
```

Linux:

```bash
python -m chess_agent.rl.report_experiment analysis/experiments --force
```

특정 실험만 보고 싶으면 마지막 인자를 해당 실험 폴더로 바꿉니다. 생성 파일은 다음과
같습니다.

- `summary.txt`: 설정, W/D/L, 평가 추이, PPO 진단과 자동 경고
- `learning_curves.png`: 평가 점수율, KL/clipping, Critic, entropy, rollout 진단 곡선
- `game_outcomes.png`: 평가 W/D/L과 학습 대국 종료 원인

그래프가 필요 없으면 `--no-plots`, 다른 위치에 저장하려면 `--output-dir`을 사용합니다.
자동 경고의 임계값은 smoke run을 빠르게 점검하기 위한 경험적 기준이며 절대적인
성공 또는 실패 판정은 아닙니다.

## Full-Chess Maskable PPO

PPO의 clipping, GAE, minibatch update는 직접 구현하지 않고 `sb3-contrib`의
`MaskablePPO`를 사용합니다. 환경의 `action_masks()`가 불법 수를 제거하며,
custom policy가 프로젝트의 residual CNN과 Policy-Value head를 연결합니다.
공식 사용법은 [Maskable PPO 문서](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)를 참고합니다.

학습을 시작하기 전에 같은 평가 설정으로 `step=0` 대국을 먼저 실행합니다. 이 값이
pretrained checkpoint의 baseline이며, 이후 평가는 같은 seed와 색 순서를 사용합니다.
학습 결과가 baseline을 넘지 못하면 `--best-model-path`에는 step 0 모델이 유지됩니다.
resume 학습에서는 step 0 대신 checkpoint의 누적 timestep에서 baseline을 기록합니다.
`--initial-model-path`에는 update 전 모델을 별도 저장하므로 학습 후에도 initial, best,
final 세 모델을 같은 조건으로 독립 평가할 수 있습니다.

`--target-kl`은 PPO update 중 policy 변화가 너무 커졌을 때 현재 rollout의 남은 epoch를
조기 종료합니다. 전체 학습을 끝내는 옵션은 아닙니다. 기본값은 `0.03`이며 안전장치를
끄려면 `--no-target-kl`, step 0 평가를 끄려면 `--no-initial-evaluation`을 사용합니다.

PPO에서는 rollout을 모을 때와 update할 때 같은 weight가 같은 action probability를
출력해야 합니다. 일반적인 train mode의 Dropout과 BatchNorm은 이 조건을 깨뜨리므로
PPO policy에서는 Dropout을 `0`으로 고정하고 BatchNorm running statistics를 업데이트하지
않습니다. convolution, linear, BatchNorm의 affine weight와 bias는 계속 gradient로
학습됩니다. Tactical supervised 모델의 Dropout은 그대로 두며, PPO 모델로 복사한 뒤에만
이 규칙을 적용합니다. `--dropout`에 0이 아닌 값을 주면 학습 시작 전에 오류를 냅니다.

같은 seed의 `step=0`, 주기 evaluation만 best checkpoint 선택에 사용합니다. 다른 seed를
사용하는 `final_evaluation`은 독립 확인용이며 점수율이 더 높아도 best 모델을 덮어쓰지
않습니다.

다음 A/B 실험은 learning rate 외의 조건을 모두 같게 두고 각각 새 pretrained model에서
시작합니다. 주기 평가는 checkpoint 선택의 표본 변동을 줄이기 위해 200판을 사용합니다.

### A: Learning rate 3e-5

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value.pt --total-timesteps 4096 --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 2048 --evaluation-games 200 --checkpoint-every 2048 --device cuda --initial-model-path tmp\full_chess_ppo_ab_lr3e5_initial.zip --save-path tmp\full_chess_ppo_ab_lr3e5_final.zip --best-model-path tmp\full_chess_ppo_ab_lr3e5_best.zip --checkpoint-dir tmp\full_chess_ppo_ab_lr3e5_checkpoints --experiment-name ppo_ab_lr3e5
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value.pt --total-timesteps 4096 --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 2048 --evaluation-games 200 --checkpoint-every 2048 --device cuda --initial-model-path tmp/full_chess_ppo_ab_lr3e5_initial.zip --save-path tmp/full_chess_ppo_ab_lr3e5_final.zip --best-model-path tmp/full_chess_ppo_ab_lr3e5_best.zip --checkpoint-dir tmp/full_chess_ppo_ab_lr3e5_checkpoints --experiment-name ppo_ab_lr3e5
```

### B: Learning rate 1e-4

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value.pt --total-timesteps 4096 --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.0001 --target-kl 0.03 --max-plies 100 --evaluation-every 2048 --evaluation-games 200 --checkpoint-every 2048 --device cuda --initial-model-path tmp\full_chess_ppo_ab_lr1e4_initial.zip --save-path tmp\full_chess_ppo_ab_lr1e4_final.zip --best-model-path tmp\full_chess_ppo_ab_lr1e4_best.zip --checkpoint-dir tmp\full_chess_ppo_ab_lr1e4_checkpoints --experiment-name ppo_ab_lr1e4
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value.pt --total-timesteps 4096 --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.0001 --target-kl 0.03 --max-plies 100 --evaluation-every 2048 --evaluation-games 200 --checkpoint-every 2048 --device cuda --initial-model-path tmp/full_chess_ppo_ab_lr1e4_initial.zip --save-path tmp/full_chess_ppo_ab_lr1e4_final.zip --best-model-path tmp/full_chess_ppo_ab_lr1e4_best.zip --checkpoint-dir tmp/full_chess_ppo_ab_lr1e4_checkpoints --experiment-name ppo_ab_lr1e4
```

4,096 timestep은 장기 성능 결론이 아니라 update 안정성과 방향을 비교하는 smoke 규모입니다.
두 실험에서 `approx_kl`, `clip_fraction`, `explained_variance`, 같은-seed 평가 점수율을
비교한 뒤 장기 학습의 learning rate를 선택합니다.

실제 500판 paired 평가에서는 `3e-5` final이 initial과 통계적으로 동률이었고,
`1e-4` final은 유의하게 낮았습니다. 따라서 다음 실험은 learning rate를 `3e-5`로
고정하고 rollout 크기만 비교합니다. 새 버전은 rollout마다 완결 대국 수, 승패 대국 수,
reward 신호 비율, return/value/advantage 표준편차를 `metrics.csv`에 기록합니다.

### Rollout 크기 A/B

두 실험은 모두 16,384 timestep을 사용합니다. A는 `1 x 256 = 256` transition마다
update하고, B는 `4 x 256 = 1,024` transition마다 update합니다. A를 다시 실행하는
이유는 이전 실험에는 새 rollout 진단값이 기록되지 않았기 때문입니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value.pt --total-timesteps 16384 --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp\full_chess_ppo_rollout256_initial.zip --save-path tmp\full_chess_ppo_rollout256_final.zip --best-model-path tmp\full_chess_ppo_rollout256_best.zip --checkpoint-dir tmp\full_chess_ppo_rollout256_checkpoints --experiment-name ppo_rollout256
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value.pt --total-timesteps 16384 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp\full_chess_ppo_rollout1024_initial.zip --save-path tmp\full_chess_ppo_rollout1024_final.zip --best-model-path tmp\full_chess_ppo_rollout1024_best.zip --checkpoint-dir tmp\full_chess_ppo_rollout1024_checkpoints --experiment-name ppo_rollout1024
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value.pt --total-timesteps 16384 --n-envs 1 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp/full_chess_ppo_rollout256_initial.zip --save-path tmp/full_chess_ppo_rollout256_final.zip --best-model-path tmp/full_chess_ppo_rollout256_best.zip --checkpoint-dir tmp/full_chess_ppo_rollout256_checkpoints --experiment-name ppo_rollout256
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value.pt --total-timesteps 16384 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --max-plies 100 --evaluation-every 8192 --evaluation-games 200 --checkpoint-every 8192 --seed 0 --device cuda --initial-model-path tmp/full_chess_ppo_rollout1024_initial.zip --save-path tmp/full_chess_ppo_rollout1024_final.zip --best-model-path tmp/full_chess_ppo_rollout1024_best.zip --checkpoint-dir tmp/full_chess_ppo_rollout1024_checkpoints --experiment-name ppo_rollout1024
```

각 학습 직후 다음 명령을 실행하면 `analysis/experiments` 바로 아래의 모든 실험에
`report/summary.txt`와 그래프가 생성됩니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.report_experiment analysis\experiments
```

Linux:

```bash
python -m chess_agent.rl.report_experiment analysis/experiments
```

두 실험에서는 평가 점수뿐 아니라 `completed_games`, `decisive_games`,
`reward_signal_rate`, `explained_variance`를 함께 비교합니다.

smoke run에서 KL과 clipping이 충분히 내려간 것을 확인한 뒤 random 상대 첫 본 학습을
실행합니다. 아직 불안정하면 본 학습으로 넘어가지 않고 `0.00001` learning rate를 같은
조건으로 비교합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp\full_chess_policy_value.pt --total-timesteps 100000 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --gamma 0.995 --entropy-coefficient 0.01 --max-plies 300 --evaluation-every 10000 --evaluation-games 200 --checkpoint-every 25000 --device cuda --initial-model-path tmp\full_chess_ppo_random_initial.zip --save-path tmp\full_chess_ppo_random_final.zip --best-model-path tmp\full_chess_ppo_random_best.zip --checkpoint-dir tmp\full_chess_ppo_random_checkpoints --experiment-dir analysis\experiments --experiment-name ppo_random_stable_v1
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --pretrained-policy-value tmp/full_chess_policy_value.pt --total-timesteps 100000 --n-envs 4 --n-steps 256 --batch-size 256 --n-epochs 2 --learning-rate 0.00003 --target-kl 0.03 --gamma 0.995 --entropy-coefficient 0.01 --max-plies 300 --evaluation-every 10000 --evaluation-games 200 --checkpoint-every 25000 --device cuda --initial-model-path tmp/full_chess_ppo_random_initial.zip --save-path tmp/full_chess_ppo_random_final.zip --best-model-path tmp/full_chess_ppo_random_best.zip --checkpoint-dir tmp/full_chess_ppo_random_checkpoints --experiment-dir analysis/experiments --experiment-name ppo_random_stable_v1
```

학습을 checkpoint에서 이어갈 때 `--total-timesteps`는 추가 학습량이 아니라 목표
누적 timestep입니다. PPO 구조와 optimizer 상태는 ZIP에서 복원됩니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_full_chess_ppo --resume-from tmp\full_chess_ppo_random_checkpoints\full_chess_ppo_100000.zip --total-timesteps 500000 --n-envs 4 --n-steps 256 --batch-size 256 --evaluation-every 10000 --evaluation-games 200 --checkpoint-every 25000 --device cuda --initial-model-path tmp\full_chess_ppo_random_resume_initial.zip --save-path tmp\full_chess_ppo_random_final.zip --best-model-path tmp\full_chess_ppo_random_best.zip --checkpoint-dir tmp\full_chess_ppo_random_checkpoints --experiment-dir analysis\experiments --experiment-name ppo_random_v1_resume
```

Linux:

```bash
python -m chess_agent.rl.train_full_chess_ppo --resume-from tmp/full_chess_ppo_random_checkpoints/full_chess_ppo_100000.zip --total-timesteps 500000 --n-envs 4 --n-steps 256 --batch-size 256 --evaluation-every 10000 --evaluation-games 200 --checkpoint-every 25000 --device cuda --initial-model-path tmp/full_chess_ppo_random_resume_initial.zip --save-path tmp/full_chess_ppo_random_final.zip --best-model-path tmp/full_chess_ppo_random_best.zip --checkpoint-dir tmp/full_chess_ppo_random_checkpoints --experiment-dir analysis/experiments --experiment-name ppo_random_v1_resume
```

독립 평가 결과는 TXT로 저장할 수 있습니다. history 길이는 checkpoint 입력 shape에서
자동으로 알아냅니다. `--output-path`를 지정하면 대국별 seed와 결과가 담긴
`<이름>_games.csv`도 자동으로 함께 생성됩니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp\full_chess_ppo_ab_lr3e5_initial.zip --games 500 --opponent random --max-plies 100 --seed 0 --device cuda --output-path analysis\ppo_ab_lr3e5_initial_500.txt
```

Linux:

```bash
python -m chess_agent.rl.evaluate_full_chess_ppo --model-path tmp/full_chess_ppo_ab_lr3e5_initial.zip --games 500 --opponent random --max-plies 100 --seed 0 --device cuda --output-path analysis/ppo_ab_lr3e5_initial_500.txt
```

best와 final도 모델 경로와 출력 이름만 바꾸고 같은 `--games`, `--seed`, `--max-plies`로
평가합니다. 두 CSV를 paired 비교하려면 다음 명령을 사용합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.compare_full_chess_evaluations analysis\ppo_ab_lr3e5_initial_500_games.csv analysis\ppo_ab_lr3e5_best_500_games.csv --output-path analysis\ppo_ab_lr3e5_initial_vs_best.txt
```

Linux:

```bash
python -m chess_agent.rl.compare_full_chess_evaluations analysis/ppo_ab_lr3e5_initial_500_games.csv analysis/ppo_ab_lr3e5_best_500_games.csv --output-path analysis/ppo_ab_lr3e5_initial_vs_best.txt
```

비교 보고서의 `Score delta`와 95% 신뢰구간은 두 번째 모델에서 첫 번째 모델을 뺀
paired 결과입니다. 신뢰구간에 0이 포함되면 관측된 차이를 확실한 향상으로 보지 않습니다.

평가 상대를 기존 tree agent로 바꾸려면 `--opponent alpha --opponent-depth 1`을
사용합니다. 학습 상대에도 같은 옵션을 쓸 수 있지만 random보다 환경 step이 훨씬
느려지므로 첫 실험은 random으로 진행합니다.

발표용으로 우선 볼 지표는 `evaluation/score_rate`, W/D/L, `average_plies`, 종료 원인,
`value_loss`, `entropy`, `approx_kl`, `clip_fraction`입니다. `policy_loss`는 부호나
절댓값이 직접 실력을 뜻하지 않으므로 단독으로 해석하지 않습니다. 특히
`max_plies` 종료가 많아 생긴 무승부 50%는 실력 향상이 아니므로 `games.csv`와 독립
평가 TXT의 `Termination breakdown`을 반드시 함께 봅니다.

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

### 약점 Theme Targeted 학습

난이도 보정 평가에서 확인한 약점 theme를 일반 sample보다 `1.5~3배` 자주 뽑는 프로필입니다.
한 sample에 여러 대상 theme가 있어도 가중치를 곱하지 않고 가장 큰 값만 적용하며, 한 epoch의 전체 sample 수는 바뀌지 않습니다.
기존 자연 분포 모델과 정확히 비교하기 위해 baseline checkpoint를 이어받지 않고 새 모델과 새 파일명으로 시작합니다.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.train_tactical_supervised --puzzles-file data\puzzle_processed\tactical_train.txt --validation-file data\puzzle_processed\tactical_valid.txt --epochs 500 --batch-size 256 --architecture cnn --hidden-size 64 --residual-blocks 3 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 --patience 15 --target-weak-themes --device cuda --save-path tmp\tactical_supervised_cnn_targeted.pt --checkpoint-path tmp\tactical_supervised_cnn_targeted_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp\tactical_supervised_cnn_targeted_best.pt
```

Linux:

```bash
python -m chess_agent.rl.train_tactical_supervised --puzzles-file data/puzzle_processed/tactical_train.txt --validation-file data/puzzle_processed/tactical_valid.txt --epochs 500 --batch-size 256 --architecture cnn --hidden-size 64 --residual-blocks 3 --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 --patience 15 --target-weak-themes --device cuda --save-path tmp/tactical_supervised_cnn_targeted.pt --checkpoint-path tmp/tactical_supervised_cnn_targeted_checkpoint.pt --checkpoint-every 1 --best-checkpoint-path tmp/tactical_supervised_cnn_targeted_best.pt
```

Windows 평가:

```powershell
.\.venv\Scripts\python -m chess_agent.rl.evaluate_tactical --agent policy --model-path tmp\tactical_supervised_cnn_targeted_best.pt --puzzles-file data\puzzle_processed\tactical_valid.txt --episodes all --device cuda --output-path analysis\tactical_evaluation_cnn_targeted.txt
```

Linux 평가:

```bash
python -m chess_agent.rl.evaluate_tactical --agent policy --model-path tmp/tactical_supervised_cnn_targeted_best.pt --puzzles-file data/puzzle_processed/tactical_valid.txt --episodes all --device cuda --output-path analysis/tactical_evaluation_cnn_targeted.txt
```

기본 프로필은 `quietMove=3`, `defensiveMove=3`, `trappedPiece=2.5`, `discoveredCheck=2.5`, `bishopEndgame=2`, `queenEndgame=2`, `advancedPawn=1.5`, `capturingDefender=1.5`, `promotion=1.5`입니다.
개별 값을 바꾸려면 `--theme-weight quietMove=4`처럼 추가합니다.
Targeted checkpoint를 재개할 때도 반드시 `--target-weak-themes`와 처음 사용한 custom weight를 동일하게 지정해야 합니다.

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

## 이전 수 History 계획

현재 observation은 한 시점의 `(18, 8, 8)` 보드 상태만 사용합니다.
캐슬링 권리와 앙파상 정보는 포함하지만 3회 반복과 50수 규칙을 정확히 판단할 history와 halfmove clock은 포함하지 않습니다.
이전 보드를 channel 방향으로 쌓으면 반복과 최근 진행뿐 아니라 수 사이의 변화를 CNN이 직접 볼 수 있습니다.

현재 Lichess tactical 파일은 첫 puzzle position 이전의 실제 대국 수순을 제공하지 않으므로 첫 agent move에 일관된 history를 만들 수 없습니다.
따라서 tactical 모델의 입력 shape을 지금 변경하지 않고, full-game/self-play 환경을 만들 때 environment가 보관한 최근 position들을 사용하는 새 observation 버전으로 추가합니다.
기존 checkpoint와 입력 shape이 달라지므로 현재 모델은 보존하고 history 모델은 별도로 관리합니다.

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
