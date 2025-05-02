import wandb
import pandas as pd
import os

project = "[TRC_torch] safety_gym TEST"
entity = "travailjade-universit-laval"
run_ids = [
   "0j2qjkco",
   "d1mr859u", 
   "q42w1l4l",
   "zfe3tvc4",
]

for run_id in run_ids:
    file_name = f"binome_run_{run_id}.csv"
    if not os.path.exists(file_name):
        print(f"⚠️ Fichier {file_name} introuvable.")
        continue

    df = pd.read_csv(file_name)

    wandb.init(project=project, entity=entity, name=f"binome_{run_id}", reinit=True)

    for _, row in df.iterrows():
        wandb.log({
            "score_log": float(row["score_log"]),
            "cost_log": float(row["cost_log"]),
            "cv_log": float(row["cv_log"])
        })

    wandb.finish()
    print(f"✅ Upload terminé pour {run_id}.")
