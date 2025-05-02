import subprocess

# Environnements pour ENTRAÎNEMENT
train_envs = [
    "SafetyPointGoal1-v0",
    "SafetyCarGoal1-v0",
    "SafetyDoggoGoal1-v0"
]

timesteps = 1_000_000
num_envs = 4

for env_id in train_envs:
    print(f"Entraînement sur {env_id}...")
    subprocess.run([
        "python", "train.py",
        "--env", env_id,
        "--timesteps", str(timesteps),
        "--num_envs", str(num_envs)
    ])
    print(f"Entraînement terminé pour {env_id}.\n")
