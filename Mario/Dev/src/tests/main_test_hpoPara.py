import sys
from pathlib import Path

# Ajoute Mario/Dev/src/ au path quel que soit l'endroit depuis lequel on lance
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mario.engine import RunEngine
from mario.algos.ppo import PPOAlgo
from mario.envs.pettingzoo_env import PettingZooEnvWrapper
from mario.algos.architectures import MLPArchitecture
from mario.hpo.spaces import PPOAlgoSpace, MLPArchiSpace

def afficher_menu():
    """Affiche les options disponibles dans la ligne de commande."""
    print("\n" + "="*60)
    print("         MARIO - ENGINE INTEGRATION TEST BENCH")
    print("="*60)
    print(" 1 [Train]        : Entraînement Classique (Standard PPO - 3 itérations)")
    print(" 2 [HPO Séquentiel] : Optimisation d'Hyperparamètres (Optuna - 5 trials, 1 worker)")
    print(" 3 [HPO Parallèle]  : Optimisation d'Hyperparamètres (Optuna - 5 trials, tous en parallèle)")
    print(" 4 [Predict]      : Test de l'inférence (.predict() avec observations)")
    print(" 5 [Render]       : Test du rendu visuel (.render() en mode human)")
    print(" 6 [Full]         : Tout exécuter à la suite (Séquence historique)")
    print(" 0 [Quitter]      : Quitter le script")
    print("="*60)
    return input("Choisissez la fonctionnalité à tester : ").strip()

def run_hpo(engine, env_mario, n_workers):
    """Lance une session HPO avec le nombre de workers spécifié."""
    mode_label = f"{n_workers} worker(s)" if n_workers != 1 else "séquentiel"
    print(f"\n>>> Démarrage du mode HPO ({mode_label})...")
    policy = engine.run_training(
        env=env_mario,
        algorithme=PPOAlgo,
        algo_hpo_space=PPOAlgoSpace(),
        archi_hpo_space=MLPArchiSpace(),
        n_trials=5,
        n_workers=n_workers,
        hpo_training_iterations=3,
        hpo_direction="maximize",
    )
    print(f"=== Optimisation HPO terminée ({mode_label}). Meilleure politique chargée. ===")
    return policy

def main():
    # Initialisation partagée des composants principaux
    engine = RunEngine()
    archi = MLPArchitecture(layers="64-64")
    env_mario = PettingZooEnvWrapper(env_name="mpe", map_name="simple_world_comm")
    
    # Variable persistante pour conserver la politique entre deux choix si nécessaire
    policy = None

    while True:
        choix = afficher_menu()

        if choix == "0":
            print("Fermeture du banc de test MARIO. À bientôt !")
            break

        elif choix == "1":
            print("\n>>> Démarrage de l'entraînement classique...")
            policy = engine.run_training(
                env=env_mario,
                algorithme=PPOAlgo,
                architecture=None,
                algo_hpo_space=None,
                archi_hpo_space=None,
                stop_criteria={"training_iteration": 3} 
            )
            print("=== Apprentissage terminé. Politique sauvegardée en mémoire. ===")

        elif choix == "2":
            # HPO séquentiel : 1 trial à la fois (comportement original)
            policy = run_hpo(engine, env_mario, n_workers=1)

        elif choix == "3":
            # HPO parallèle : autant de workers que de trials (défaut None)
            policy = run_hpo(engine, env_mario, n_workers=None)

        elif choix == "4":
            print("\n>>> Test de la méthode predict()...")
            if policy is None:
                print("[INFO] Aucune politique active trouvée. Lancement d'un entraînement flash (1 itération)...")
                policy = engine.run_training(
                    env=env_mario, algorithme=PPOAlgo, stop_criteria={"training_iteration": 1}
                )
            
            obs_dict = env_mario.reset()
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]
                
            print(f"Observations brutes reçues (clés agents) : {list(obs_dict.keys())}")
            actions = policy.predict(obs_dict)
            print(f"-> Actions calculées avec succès : {actions}")

        elif choix == "5":
            print("\n>>> Test de la méthode render()...")
            if policy is None:
                print("[INFO] Aucune politique active trouvée pour le rendu. Lancement d'un entraînement flash...")
                policy = engine.run_training(
                    env=env_mario, algorithme=PPOAlgo, stop_criteria={"training_iteration": 1}
                )
            
            print("Ouverture de la fenêtre de rendu (Gym/PettingZoo)...")
            policy.render(save_mode="human")
            print("=== Exécution du render réussi. ===")

        elif choix == "6":
            print("\n>>> Lancement de la séquence de test complète...")
            
            print("\n[Étape 1/5] Entraînement standard")
            policy = engine.run_training(
                env=env_mario, algorithme=PPOAlgo, stop_criteria={"training_iteration": 3} 
            )
            
            print("\n[Étape 2/5] Recherche HPO séquentielle")
            policy = run_hpo(engine, env_mario, n_workers=1)

            print("\n[Étape 3/5] Recherche HPO parallèle")
            policy = run_hpo(engine, env_mario, n_workers=None)

            print("\n[Étape 4/5] Inférence (Predict)")
            obs_dict = env_mario.reset()
            if isinstance(obs_dict, tuple): obs_dict = obs_dict[0]
            actions = policy.predict(obs_dict)
            print(f"Actions : {actions}")
            
            print("\n[Étape 5/5] Rendu graphique")
            policy.render(save_mode="human")
            
            print("\n=== Tout le flux opérationnel a été exécuté avec succès ! ===")

        else:
            print("[ERREUR] Option invalide. Veuillez entrer un nombre entre 0 et 6.")

if __name__ == "__main__":
    main()