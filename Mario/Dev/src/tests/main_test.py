import sys
import subprocess
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
    # --- SÉLECTION DYNAMIQUE DE L'ENVIRONNEMENT ---
    print("=== Configuration du test ===")
    print("Choix possibles :")
    print("1. mpe (Multi-Agent Particle Environments)")
    print("2. lbf (Level Based Foraging)")
    
    choix = input("Entrez votre choix (lbf ou mpe) [Défaut: mpe] : ").strip().lower()
    
    # Configuration par défaut ou application du choix utilisateur
    if choix == "lbf":
        selected_env = "lbf"
        selected_map = "Foraging-8x8-2p-2f-v2"
    else:
        selected_env = "mpe"
        selected_map = "simple_world_comm"
    print(f"-> Environnement sélectionné : {selected_env} ({selected_map})\n")
    # -----------------------------------------------
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
    env_mario = PettingZooEnvWrapper(env_name=selected_env, map_name=selected_map)

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

    # Rendu en fonction du choix d'environnement
    if choix == "lbf":
        print("Démarrage du rendu visuel interactif pour LBF...")
        try:
            import time
            import gym
            import lbforaging

            # 1. On recrée l'environnement natif pour éviter les conflits avec Ray
            raw_env = gym.make("Foraging-8x8-2p-2f-v2")
            obs = raw_env.reset()
            
            done = False
            step = 0
            
            # 2. Boucle de simulation
            while not done and step < 100:
                # Utilise ta politique déjà entraînée (policy)
                actions = policy.predict(obs) 
                
                # Step dans l'env
                obs, rewards, dones, infos = raw_env.step(actions)
                
                # Rendu Pygame
                raw_env.render(mode="human")
                time.sleep(0.1)
                
                done = all(dones.values()) if isinstance(dones, dict) else all(dones)
                step += 1
            
            raw_env.close()
            print("Rendu visuel LBF terminé.")
            
        except Exception as e:
            print(f"Erreur lors du rendu visuel : {e}")
            print("Passage au flux standard.")
    else:
        # Rendu natif pour MPE
        policy.render(save_mode="human")

    print("Execution du render réussi.")

    print("Test réussi ! Flux operational.")

if __name__ == "__main__":
    main()