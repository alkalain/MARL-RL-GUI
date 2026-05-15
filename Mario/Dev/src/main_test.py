from mario.engine import RunEngine
from mario.algos.ppo import PPOAlgo
from mario.envs.pettingzoo_env import PettingZooEnvWrapper
from mario.algos.architectures import MLPArchitecture

def main():
    """
    Script de test d'intégration pour valider le flux de travail de MARIO.
    
    Cette fonction illustre l'assemblage complet des composants :
    1. Initialisation du moteur d'exécution.
    2. Configuration de l'algorithme et de son architecture.
    3. Définition de l'environnement de simulation.
    4. Lancement de la procédure d'apprentissage.
    """
    # 1. Instanciation du moteur (coordinateur de la session)
    engine = RunEngine()

    # 2. Configuration de l'algorithme (Adaptateur MARLlib)
    # Définition d'une architecture MLP avec deux couches cachées de 128 neurones
    archi = MLPArchitecture(layers="128-128")
    ppo = PPOAlgo(
        architecture=archi,
        )

    # 3. Préparation de l'environnement via le wrapper MARIO
    # On utilise ici un scénario multi-agent de type 'MPE' (Multi-Agent Particle Environments)
    env_mario = PettingZooEnvWrapper(env_name="mpe", map_name="simple_world_comm")

    # 4. Exécution du processus d'apprentissage
    # Le moteur orchestre l'appel à la méthode de train de l'algorithme choisi
    print("Démarrage du test d'intégration...")
    policy = engine.run_training(
        env=env_mario,
        algorithme=PPOAlgo,
        architecture=archi,
        algo_hpo_space=None, # Pas encore utilisé
        archi_hpo_space=None # Pas encore utilisé
    )

    policy.render(
        #env=env_mario,
        #model=policy.model,
        save_mode="human")

    print("Test réussi ! Politique générée.")

if __name__ == "__main__":
    main()