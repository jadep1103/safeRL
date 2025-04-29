import wandb

api = wandb.Api()

PROJECT = "safe-rl"

# On récupère tous les runs avec tag 'test'
testing_runs = [run for run in api.runs(f"{wandb.run.entity}/{PROJECT}") if 'test' in (run.tags or [])]

report = wandb.Report(
    project=PROJECT,
    title="Testing Metrics Dashboard",
    description="Testing curves: reward, cost, cost per step, episode length"
)

for run in testing_runs:
    report.add_line(
        title=f"[Testing] {run.config['env']} - Total Reward",
        x="episode",
        y="score_log",
        run=run.id
    )
    report.add_line(
        title=f"[Testing] {run.config['env']} - Total Cost",
        x="episode",
        y="cost_log",
        run=run.id
    )
    report.add_line(
        title=f"[Testing] {run.config['env']} - Cost per Step",
        x="episode",
        y="cv_log",
        run=run.id
    )
    report.add_line(
        title=f"[Testing] {run.config['env']} - Episode Length",
        x="episode",
        y="episode_length",
        run=run.id
    )

report.save()

print(f"Dashboard TESTING créé : {report.url}")

