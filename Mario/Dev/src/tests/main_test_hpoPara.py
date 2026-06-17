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
    print("         MARIO - ENGINE INTEGRATION TEST BENCH (HPO)")
    print("="*60)
    print(" 1 [Train]          : Entraînement Classique (Standard PPO - 3 itérations)")
    print(" 2 [HPO Séquentiel] : Optimisation d'Hyperparamètres (Optuna - 5 trials, 1 worker)")
    print(" 3 [HPO Parallèle]  : Optimisation d'Hyperparamètres (Optuna - 5 trials, tous en parallèle)")
    print(" 4 [Render]         : Test du rendu visuel (human / gif / video, choix run+checkpoint)")
    print(" 5 [Full]           : Tout exécuter à la suite (Séquence historique)")
    print(" 0 [Quitter]        : Quitter le script")
    print("="*60)
    return input("Choisissez la fonctionnalité à tester : ").strip()


def afficher_menu_env():
    """Sous-menu pour choisir dynamiquement l'environnement, à l'image de main_test.py."""
    print("\n=== Configuration du test ===")
    print("Choix possibles :")
    print("1. mpe (Multi-Agent Particle Environments)")
    print("2. lbf (Level Based Foraging)")
    choix_env = input("Entrez votre choix (lbf ou mpe) [Défaut: mpe] : ").strip().lower()

    if choix_env == "lbf":
        return "lbf", "Foraging-8x8-2p-2f-v2"
    return "mpe", "simple_world_comm"


def afficher_menu_render():
    """Sous-menu pour choisir le mode de rendu."""
    print("\n" + "-"*40)
    print("  Mode de rendu :")
    print("   1) human  (fenêtre temps réel)")
    print("   2) gif    (export .gif)")
    print("   3) video  (export .mp4)")
    print("-"*40)
    choix = input("Choisir le mode [1-3] (défaut: 1) : ").strip()
    return {"1": "human", "2": "gif", "3": "video", "": "human"}.get(choix, "human")


def afficher_menu_checkpoint():
    """Sous-menu pour choisir le critère de sélection du checkpoint."""
    print("\n" + "-"*40)
    print("  Sélection du checkpoint :")
    print("   1) Choix interactif (liste des checkpoints du run)")
    print("   2) last  (dernier checkpoint)")
    print("   3) best  (meilleur episode_reward_mean)")
    print("-"*40)
    choix = input("Choisir [1-3] (défaut: 1) : ").strip()
    return {"1": None, "2": "last", "3": "best", "": None}.get(choix, None)


def run_render(policy):
    """Pilote l'appel à policy.render() avec les options choisies par l'utilisateur."""
    save_mode = afficher_menu_render()
    checkpoint = afficher_menu_checkpoint()

    print(f"\n>>> Lancement du rendu (save_mode='{save_mode}', checkpoint={checkpoint!r})...")
    policy.render(save_mode=save_mode, checkpoint=checkpoint)
    print("=== Exécution du render réussie. ===")


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

    # --- SÉLECTION DYNAMIQUE DE L'ENVIRONNEMENT (comme dans main_test.py) ---
    selected_env, selected_map = afficher_menu_env()
    print(f"-> Environnement sélectionné : {selected_env} ({selected_map})\n")

    env_mario = PettingZooEnvWrapper(env_name=selected_env, map_name=selected_map)

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
                architecture=archi,
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
            print("\n>>> Test de la méthode render()...")
            if policy is None:
                print("[INFO] Aucune politique active trouvée pour le rendu. Lancement d'un entraînement flash...")
                policy = engine.run_training(
                    env=env_mario, algorithme=PPOAlgo, architecture=archi,
                    stop_criteria={"training_iteration": 1}
                )

            run_render(policy)

        elif choix == "5":
            print("\n>>> Lancement de la séquence de test complète...")

            print("\n[Étape 1/4] Entraînement standard")
            policy = engine.run_training(
                env=env_mario, algorithme=PPOAlgo, architecture=archi,
                stop_criteria={"training_iteration": 3}
            )

            print("\n[Étape 2/4] Recherche HPO séquentielle")
            policy = run_hpo(engine, env_mario, n_workers=1)

            print("\n[Étape 3/4] Recherche HPO parallèle")
            policy = run_hpo(engine, env_mario, n_workers=None)

            print("\n[Étape 4/4] Rendu graphique")
            run_render(policy)

            print("\n=== Tout le flux opérationnel a été exécuté avec succès ! ===")

        else:
            print("[ERREUR] Option invalide. Veuillez entrer un nombre entre 0 et 5.")


if __name__ == "__main__":
    main()