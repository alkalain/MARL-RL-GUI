from mario.engine import RunEngine
from mario.algos.ppo import PPOAlgo
from mario.envs.pettingzoo_env import PettingZooEnvWrapper

def main():
    # 1. On prépare le moteur (le cerveau qui gère la session)
    engine = RunEngine()

    # 2. On définit l'algo (ton wrapper MARLlib)
    # On peut lui passer les hyperparamètres ici
    ppo = PPOAlgo(hyperparams={
        "core_arch": "mlp", 
        "encode_layer": "128-128"
    })

    # 3. On définit l'environnement (ton wrapper Mario)
    # On lui donne les noms que MARLlib comprend (ex: mpe / simple_spread)
    env_mario = PettingZooEnvWrapper(env_name="mpe", map_name="simple_spread")

    # 4. On lance la machine !
    # Le moteur va appeler ppo.train(env_name="mpe", ...)
    print("Démarrage du test d'intégration...")
    policy = engine.run_training(
        env=env_mario,
        algo=ppo,
        architecture=None, # Pas encore utilisé
        algo_hpo_space=None, # Pas encore utilisé
        archi_hpo_space=None # Pas encore utilisé
    )

    print("Test réussi ! Politique générée.")

if __name__ == "__main__":
    main()