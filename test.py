import gymnasium 

# Liste des environnements disponibles dans Gymnasium
envs = gymnasium.envs.registration.registry
for env_id in envs.keys():
    print(env_id)
