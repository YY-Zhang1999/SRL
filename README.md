srlnbc/
├── README.md
├── requirements.txt
├── setup.py
├── srlnbc/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py      # 基础智能体类
│   │   └── srlnbc_agent.py    # 主要的SRLNBC智能体实现
│   ├── models/
│   │   ├── __init__.py
│   │   ├── barrier.py         # Neural Barrier Certificate网络
│   │   ├── policy.py          # 策略网络
│   │   └── value.py           # 价值网络
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── buffer.py          # 经验回放缓冲区
│   │   ├── logger.py          # 日志工具
│   │   └── scheduler.py       # 学习率调度器等
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── barrier_loss.py    # Barrier Certificate的损失函数
│   │   ├── policy_loss.py     # 策略网络的损失函数  
│   │   └── value_loss.py      # 价值网络的损失函数
│   └── configs/
│       ├── __init__.py
│       └── default_configs.py  # 默认配置参数
├── examples/
│   ├── __init__.py
│   ├── classic_control.py     # 经典控制示例
│   ├── safety_gym.py          # Safety Gym环境示例
│   └── metadrive.py           # MetaDrive环境示例
└── tests/
    ├── __init__.py
    ├── test_barrier.py        # Barrier相关测试
    ├── test_policy.py         # 策略相关测试
    └── test_agent.py          # 智能体相关测试

