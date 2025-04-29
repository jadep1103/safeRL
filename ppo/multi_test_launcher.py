import subprocess

# Environnements pour TEST
test_envs = [
    "SafetyPointGoal1-v0",
    "SafetyPointGoal2-v0",
    "SafetyCarGoal1-v0",
    "SafetyCarGoal2-v0",
    "SafetyDoggoGoal1-v0"  # Pas de Goal2 pour Doggo
]

episodes = 5

for env_id in test_envs:
    print(f"Test sur {env_id}...")
    subprocess.run([
        "python", "test.py",
        "--env", env_id,
        "--episodes", str(episodes)
    ])
    print(f"Test terminé pour {env_id}.\n")
