import wandb

api = wandb.Api()

PROJECT = "safe-rl"

# On récupère tous les runs avec tag 'train'
training_runs = [run for run in api.runs(f"{wandb.run.entity}/{PROJECT}") if 'train' in (run.tags or [])]

report = wandb.Report(
    project=PROJECT,
    title="Training Metrics Dashboard",
    description="Training curves: reward, cost, cost per step, episode length"
)

# Ajoute les métriques par environnement
for run in training_runs:
    report.add_line(
        title=f"[Training] {run.config['env']} - Total Reward",
        x="timesteps",
        y="score_log",
        run=run.id
    )
    report.add_line(
        title=f"[Training] {run.config['env']} - Total Cost",
        x="timesteps",
        y="cost_log",
        run=run.id
    )
    report.add_line(
        title=f"[Training] {run.config['env']} - Cost per Step",
        x="timesteps",
        y="cv_log",
        run=run.id
    )
    report.add_line(
        title=f"[Training] {run.config['env']} - Episode Length",
        x="timesteps",
        y="episode_length",
        run=run.id
    )

report.save()

print(f"Dashboard TRAINING créé : {report.url}")
