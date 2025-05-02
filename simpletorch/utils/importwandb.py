import wandb
import pandas as pd
import os

# Config binôme
entity = "estellou590-university-of-laval"
project = "safe-rl"
run_ids = [  # 👉 remplace par les vrais IDs
    "0j2qjkco",
    "d1mr859u",  # exemple
    "q42w1l4l",
    "zfe3tvc4",
]

api = wandb.Api()

for run_id in run_ids:
    run = api.run(f"{entity}/{project}/{run_id}")
    df = run.history()
    df = df[["episode", "episode_length", "score_log", "cost_log", "cv_log"]].dropna()
    df.to_csv(f"binome_run_{run_id}.csv", index=False)
    print(f"✅ Run {run_id} téléchargé.")
