# Safe Reinforcement Learning Through Neural Barrier Certificate

This repository implements the paper "Model-Free Safe Reinforcement Learning Through Neural Barrier Certificate" using PyTorch and Stable-Baselines3.

## Features

- Implementation of Safe TD3 with Neural Barrier Certificates
- Support for both standard Gym and Safety Gym environments  
- Extensive testing suite and benchmarking tools
- Easy integration with Stable-Baselines3

## Installation

1. Create a virtual environment:
```bash
conda create -n safe-rl python=3.8
conda activate safe-rl
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install Safety Gym (optional, for safety environments):
```bash
pip install safety-gym
```

## Project Structure

```
SRL/
├── README.md
├── requirements.txt
├── setup.py
├── core/
│   ├── agents/
│   │   ├── base_agent.py      # Base agent class
│   │   └── TD3_agent.py       # Safe TD3 implementation
│   ├── models/
│   │   ├── barrier.py         # Neural Barrier Certificate network
│   │   ├── policy.py          # Policy network
│   │   └── value.py           # Value network
│   ├── utils/
│   │   ├── buffer.py          # Experience replay buffer
│   │   ├── logger.py          # Logging utilities
│   │   └── scheduler.py       # Learning rate scheduler
│   ├── losses/
│   │   ├── barrier_loss.py    # Barrier Certificate loss
│   │   ├── policy_loss.py     # Policy network loss
│   │   └── value_loss.py      # Value network loss
│   └── configs/
│       └── default_configs.py  # Default configurations
├── examples/
│   ├── classic_control.py     # Classic control examples
│   ├── safety_gym.py          # Safety Gym examples
│   └── metadrive.py          # MetaDrive examples
└── tests/
    ├── test_barrier.py        # Barrier network tests
    ├── test_policy.py         # Policy tests
    └── test_agent.py          # Agent tests
```

## Usage

1. Basic training example:

```python
from srlnbc.agents.TD3_agent import TD3SafeAgent

# Create agent
agent = TD3SafeAgent(
    policy="SafeTD3Policy",
    env=env,
    learning_rate=3e-4,
    barrier_lambda=0.1,
    n_barrier_steps=20,
    gamma_barrier=0.99,
    safety_margin=0.1
)

# Train agent
agent.learn(total_timesteps=1_000_000)
```

2. Run benchmarks:

```bash
python examples/benchmark_td3.py \
    --seed 0 \
    --total_timesteps 1000000 \
    --eval_freq 10000
```

3. Run tests:

```bash
python -m unittest tests/test_barrier.py
python -m unittest tests/test_policy.py
python -m unittest tests/test_agent.py
```

## Experiments

The implementation includes experiments on:

1. Classic Control Tasks:
- Emergency Braking Control
- Reach-Avoid Control

2. Safety Gym Tasks:
- Point Goal
- Car Goal
- Doggo Goal

3. MetaDrive Tasks:
- Safe Navigation
- Multi-Agent Scenarios

## Results

Example results comparing Safe TD3 with baseline TD3:

1. Safety Gym Performance:
- Near-zero constraint violations
- Comparable or higher rewards
- Successful learning of feasible regions

2. Classic Control:
- Emergency Braking: Safe stopping with minimum distance
- Reach-Avoid: Successful goal reaching while avoiding obstacles

3. MetaDrive:
- High success rate in navigation
- Significant reduction in collision rates

## Citing

If you use this code in your research, please cite:

```bibtex
@article{yang2023model,
  title={Model-Free Safe Reinforcement Learning Through Neural Barrier Certificate},
  author={Yang, Yujie and Jiang, Yuxuan and Liu, Yichen and Chen, Jianyu and Li, Shengbo Eben},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={3},
  pages={1295--1302},
  year={2023},
  publisher={IEEE}
}
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.