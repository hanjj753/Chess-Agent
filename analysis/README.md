# 분석 결과 구조

실험 결과는 주제별 폴더에 저장합니다. 새 평가 명령을 실행할 때도 아래 폴더를
`--output-path`로 사용하면 루트가 다시 복잡해지지 않습니다.

| 폴더 | 내용 |
| --- | --- |
| `tactical/` | Tactical supervised 모델의 퍼즐 평가 |
| `value/` | Value head 단독 평가 |
| `ppo_baseline/` | 초기 PPO, learning rate, mode fix 비교 |
| `ppo_value/` | Rollout 크기와 value pretraining 비교 |
| `ppo_alpha_p10/` | Alpha-random p10 curriculum 평가 |
| `ppo_reward_shaping/` | p25 환경의 terminal 처리와 reward shaping 비교 |
| `ppo_entropy/` | Entropy coefficient 대조 실험과 policy drift |
| `ppo_horizon/` | `max_plies=100/200` 평가 및 학습 horizon 비교 |
| `experiments/` | 학습 run별 config, metrics, games, summary와 자동 보고서 |
| `losses_1600/`, `losses_2000/` | Stockfish 상대 패배 PGN과 분석 결과 |

각 독립 대국 평가의 `.txt`와 `_games.csv`는 같은 폴더에 둡니다. 두 모델을 비교한
보고서도 원본 평가 파일과 같은 주제 폴더에 저장합니다. Policy drift 평가는 요약
`.txt`와 position별 `_positions.csv`를 함께 보관합니다.

## 현재 Entropy 실험 결론

`entropy_coefficient=0.01`과 `0`은 세 학습 seed 모두 사전학습 policy의 tactical
top-1 선택을 98% 이상 유지했습니다. 독립 1,000판 평가에서도 두 설정 사이의 차이는
일관되거나 통계적으로 유의하지 않았습니다. 따라서 entropy가 PPO 성능 정체의 주된
원인이라는 가설은 지지되지 않으며, 현재 기본값 `0.01`을 유지합니다.

## 현재 Horizon 실험 결론

평가 horizon을 100 ply에서 200 ply로 늘리자 인위적인 `max_plies` 무승부가 약 80%
줄고 모든 모델의 점수가 약 5%p 상승했습니다. 따라서 독립 평가는 200 ply를 기준으로
합니다. 200-ply 학습 모델은 기존 100-ply 학습 모델보다 세 seed 평균 +0.78%p였지만,
seed별 차이가 -2.00/+2.70/+1.65%p로 커서 아직 개선으로 확정하지 않습니다. Seed 3~5의
100/200-ply paired 학습으로 표본을 늘린 뒤 학습 horizon을 확정합니다.
