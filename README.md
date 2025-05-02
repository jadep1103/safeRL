# Safe RL Project – PPO & TRC

Ce projet a été réalisé dans le cadre du cours IFT-7201 (Apprentissage par renforcement) à l'Université Laval. Il s'appuie principalement sur l'article suivant :

> **TRC: Trust Region Conditional Value at Risk for Safe Reinforcement Learning** — Dohyeong Kim and Songhwai Oh, *IEEE Robotics and Automation Letters*, 2022.

Nous avons comparé deux approches d’apprentissage par renforcement :  
- **PPO (Proximal Policy Optimization)** : méthode classique sans contrainte,  
- **TRC (Trust Region Conditional Value at Risk)** : méthode sécuritaire.

Les agents ont été évalués dans des environnements vectorisés de [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/en/latest/).

---

## 👤 Auteurs

- **Jade Piller-Cammal**  
  NI : 537 306 695
  📧 jade.piller-cammal.1@ulaval.ca

- **Estelle Tournassat**  
  NI : 537 305 301  
  📧 estelle.tournassat.1@ulaval.ca

---

## 🔧 Installation

### Option 1 — Environnement Python

Créer un environnement virtuel compatible et installer les dépendances :

```bash
pip install -r requirements.txt
```

> ⚠️ Python 3.10 recommandé pour la compatibilité avec certaines bibliothèques.

---

### Option 2 — Docker

Le projet peut être lancé via le Dockerfile fourni :

```bash
docker build -t safe-rl .
docker run --rm -it safe-rl
```

---

## Lancement des agents (des exemples de lancement sont inclus dans le code)

### PPO

```bash
python ppo/train.py
```

### TRC

#### Entraînement (ex: SafetyPointGoal1)

```bash
python trc/main.py --env_name SafetyPointGoal1-v0 --n_envs 4 --name TRC_PointGoal1 --wandb
```

#### Test

```bash
python trc/main.py --env_name SafetyPointGoal1-v0 --n_envs 1 --test --name TRC_PointGoal1 --wandb
```

Tous les logs sont automatiquement envoyés vers [Weights & Biases](https://wandb.ai/) si l’option `--wandb` est activée.

---

## 📂 Structure du dépôt

```
safeRL/
├── ppo/                ← Implémentation PPO (baseline)
├── trc/                ← Implémentation TRC (Trust Region CVaR)
├── Dockerfile
└── requirements.txt      
```

---

## 📄 Licence

Le code sur lequel est basé ce projet a été distribué sous licence MIT par les auteurs de l'article.
