# 🚀 SafeRL: Modernized Safe Reinforcement Learning Implementation

A modern reimplementation of the **Trust Region Conditional Value at Risk (TRC)** algorithm using updated libraries and compared against **Proximal Policy Optimization (PPO)** as a baseline. This project updates the original implementation from older MuJoCo versions to modern mujocopy &`safety-gymnasium` environments.

> **Based on:** *TRC: Trust Region Conditional Value at Risk for Safe Reinforcement Learning*  
> Dohyeong Kim and Songhwai Oh, *IEEE Robotics and Automation Letters*, 2022.

## 🎯 What is SafeRL?

This project demonstrates **safe reinforcement learning** - teaching AI agents to achieve goals while respecting safety constraints. Unlike standard RL that only maximizes rewards, safe RL also minimizes constraint violations (like collisions or unsafe behaviors).

### 🔬 Algorithms Compared

| Algorithm | Type | Key Features |
|-----------|------|-------------|
| **PPO** | Standard RL | Fast, stable, but ignores safety constraints |
| **TRC** | Safe RL | Uses Conditional Value at Risk (CVaR) to ensure safety |

### 🏟️ Environments

The agents are tested on **Safety-Gymnasium** environments:
- **SafetyPointGoal1/2-v0**: Point robot navigation with obstacles
- **SafetyCarGoal1/2-v0**: Car navigation avoiding hazards  
- **SafetyDoggoGoal1-v0**: Quadruped robot locomotion

### 🔄 Sim-to-Real Transfer Experiments

A key experimental focus was exploring **sim-to-real transfer** in safe RL by treating different Safety-Gymnasium difficulty levels as "simulation" vs "real" environments:

- **"Simulation"**: Goal1 environments (easier safety constraints)
- **"Real World"**: Goal2 environments (harder safety constraints, more obstacles)

This setup allowed us to study how well safety policies trained in easier conditions transfer to more challenging scenarios - a critical question for deploying safe RL in real-world applications.

---

## 🛠️ Quick Setup

### Prerequisites
- Python 3.10+ (recommended for compatibility)
- GPU optional but recommended for training

### Option 1: Local Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd safeRL-1

# Create virtual environment
python -m venv saferl-env
source saferl-env/bin/activate  # or `saferl-env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker (may be tricky for env render)
```bash
docker build -t safe-rl .
docker run --rm -it safe-rl
```

---

## 🚀 Usage

### Training Agents

#### PPO (Baseline)
```bash
# Train PPO on car environment
python ppo/train.py --env SafetyCarGoal1-v0 --timesteps 1000000 --num_envs 4

# Available environments
python ppo/train.py --env SafetyPointGoal1-v0
python ppo/train.py --env SafetyDoggoGoal1-v0
```

#### TRC (Safe RL)
```bash
# Train TRC with safety constraints
python trc/main.py --env_name SafetyPointGoal1-v0 --n_envs 1 --name TRC_Point --wandb

# Training with multiple environments (recommended)
python trc/main_multi.py --env_name SafetyCarGoal1-v0 --n_envs 4 --name TRC_Car_Multi --wandb
```

### Testing Trained Models

#### PPO Testing
```bash
python ppo/test.py --env SafetyCarGoal1-v0 --episodes 10
```

#### TRC Testing  
```bash
python trc/main.py --env_name SafetyPointGoal1-v0 --test --name TRC_Point_Test --wandb
```

### Sim-to-Real Transfer Testing
```bash
# Train on "simulation" (Goal1 - easier)
python trc/main.py --env_name SafetyPointGoal1-v0 --name TRC_Point_Sim --wandb

# Test on "real world" (Goal2 - harder) 
python trc/main.py --env_name SafetyPointGoal2-v0 --test --name TRC_Point_Transfer --wandb

# Compare performance and safety violations between domains
```

### Batch Training/Testing
```bash
# Train PPO on all environments
python ppo/multi_train_launcher.py

# Test all environments
python ppo/multi_test_launcher.py
```

---

## 📊 Monitoring & Visualization

### Weights & Biases Integration
Enable W&B logging with `--wandb` flag to track:
- **Reward/Score**: Task performance
- **Cost/Constraint Violations**: Safety metrics  
- **CVaR**: Conditional Value at Risk for tail risk
- **Training Loss**: Policy and value function losses

---

## 🏗️ Project Structure

```
safeRL-1/
├── 📁 ppo/                          # PPO implementation (baseline)
│   ├── train.py                     # Main training script
│   ├── test.py                      # Testing trained models
│   ├── wrapper.py                   # Safety-Gymnasium compatibility
│   ├── multi_train_launcher.py      # Batch training
│   ├── multi_test_launcher.py       # Batch testing  
│   └── models/                      # Saved PPO models
│       ├── SafetyCarGoal1-v0/
│       ├── SafetyPointGoal1-v0/
│       └── ...
│
├── 📁 trc/                          # TRC implementation (safe RL)
│   ├── main.py                      # Training/testing (single env)
│   ├── main_multi.py                # Multi-environment training
│   ├── agent.py                     # TRC agent implementation
│   ├── models.py                    # Neural network architectures
│   ├── logger.py                    # Training metrics logging
│   ├── utils/                       # Utility scripts
│   ├── bash_files/                  # Training scripts
│   └── results/                     # TRC training results
│       ├── TRC_FINAL_car_s42/
│       ├── TRC_FINAL_point_s42/
│       └── ...
│
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container setup
└── README.md                        # This file
```

---

## ⚙️ Key Parameters

### TRC Algorithm
- `--cost_d`: Safety constraint threshold (default: 0.025)
- `--cost_alpha`: CVaR confidence level (default: 0.125)  
- `--max_kl`: Trust region constraint (default: 0.001)
- `--n_epochs`: Policy update epochs (default: 200)

### Training
- `--total_steps`: Total training steps (default: 10M)
- `--n_envs`: Parallel environments (default: 1 for TRC, 4 for PPO)
- `--seed`: Random seed for reproducibility

---

## 🔬 Results & Analysis

The project includes pre-trained models and results in:
- `ppo/models/`: Trained PPO agents
- `trc/results/`: TRC training logs and checkpoints

### Key Metrics to Compare:
1. **Episode Return**: Total reward achieved
2. **Constraint Violations**: Safety performance  
3. **Training Stability**: Convergence behavior
4. **Sample Efficiency**: Steps needed to learn
5. **Transfer Performance**: How well Goal1→Goal2 policies transfer
6. **Safety Degradation**: Constraint violation increase during transfer

---

## 💡 Fun Ideas for Extension

Project was mostly academic but could be extended for:

1. **Domain Adaptation**: Extend sim-to-real transfer to actual physical robots ?
3. **Custom Environments**: Create your own safety-critical scenarios
4. **Hyperparameter Tuning**: Optimize safety vs. performance trade-offs
5. **Visualization**: Create cool demos of safe vs. unsafe behaviors
6. **Transfer Learning**: Study how different environment pairs affect transfer success


---

## 🐛 Known Issues

- TRC currently works best with `n_envs=1` (single environment)
- GPU memory requirements can be high for large batch sizes

---

## 📚 References

- [TRC Paper](https://ieeexplore.ieee.org/document/9830207) - Original TRC algorithm
- [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/) - Modern safety RL environments  
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation

---

## 👥 Contributors

- **Jade Piller-Cammal**
- **Estelle Tournassat** 

*Originally developed for IFT-7201 (Reinforcement Learning) at Université Laval*

---

## 📄 License

MIT License
