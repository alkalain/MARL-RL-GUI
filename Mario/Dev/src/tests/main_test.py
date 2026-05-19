import sys
from pathlib import Path

# Ajoute Mario/Dev/src/ au path quel que soit l'endroit depuis lequel on lance
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    # Définition d'une architecture MLP avec deux couches cachées de 64 neurones pour le test
    archi = MLPArchitecture(layers="64-64")
    ppo = PPOAlgo(
        architecture=archi,
        )

    # 3. Préparation de l'environnement via le wrapper MARIO
    # On utilise ici un scénario simple 'simple_v3' pour accélérer les tests
    env_mario = PettingZooEnvWrapper(env_name="mpe", map_name="simple_world_comm")

    # 4. Exécution du processus d'apprentissage
    # Le moteur orchestre l'appel à la méthode de train de l'algorithme choisi
    print("Démarrage du test d'intégration...")
    policy = engine.run_training(
        env=env_mario,
        algorithme=PPOAlgo,
        architecture=None, # Configuration optionnelle si déjà définie dans l'objet ppo
        algo_hpo_space=None, # Emplacement réservé pour l'optimisation future
        archi_hpo_space=None, # Emplacement réservé pour l'optimisation future
        stop_criteria={"training_iteration": 3} 
    )

    print("Apprentissage terminé. Politique générée.")

    # 5. Test du chargement de checkpoint et rendu
    # Simulation du chargement d'un état sauvegardé pour le rendu futur
    checkpoint_path = None  # À remplir avec le chemin local vers le checkpoint Ray
    
    if checkpoint_path:
        print(f"Chargement du checkpoint : {checkpoint_path}")
        # policy = engine.load_policy(algo=ppo, path=checkpoint_path)
    
    # Vérification du moteur d'inférence (Predict)
    # On s'assure que la structure de sortie est correcte même avec le placeholder
    print("Test de l'inférence (predict)...")
    obs = {agent: None for agent in ["agent_0"]} # Format d'entrée fictif
    actions = policy.predict(obs)
    print(f"Actions calculées : {actions}")

    policy.render(
        #env=env_mario,
        #model=policy.model,
        save_mode="human")

    print("Execution du render réussi.")

    print("Test réussi ! Flux opérationnel.")

if __name__ == "__main__":
    main()