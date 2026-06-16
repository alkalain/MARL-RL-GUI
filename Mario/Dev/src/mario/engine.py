import os
from typing import Optional

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario.envs.base import MultiAgentEnv
from mario.algos.base import Algo, ArchitectureSupport
from mario.algos.policies import JointPolicy
from mario.hpo.spaces import AlgoHyperparametersResearchSpace, ArchiHyperparametersResearchSpace
from mario.utils.stats import Stats
from marllib import marl
from typing import Union

class RunEngine:
    """
    Moteur d'exécution principal chargé de coordonner le cycle de vie des processus.
    
    Cette classe centralise la logique de pilotage en orchestrant les interactions 
    entre l'environnement, l'algorithme d'apprentissage et les structures de recherche 
    d'hyperparamètres.
    """
    def __init__(self, engine_type: str = "default"):
        """
        Initialise l'instance du moteur d'exécution.

        Args:
            engine_type (str): Identifiant du moteur (défaut : "default").
        """
        self.type = engine_type

    def run_training(
        self,
        env: MultiAgentEnv,
        algorithme: Union[type, Algo],
        architecture: ArchitectureSupport = None,
        algo_hpo_space: AlgoHyperparametersResearchSpace = None,
        archi_hpo_space: ArchiHyperparametersResearchSpace = None,
        stop_criteria: dict = None,
        GPUs=0,
        Checkpoints_freq=1,
        n_trials: int = 10,
        n_workers: Optional[int] = None,
        hpo_training_iterations: int = 5,
        hpo_direction: str = "maximize",
    ) -> JointPolicy:
        """
        Pilote une session complète d'entraînement.
        
        Cette méthode initialise l'algorithme s'il ne l'est pas encore, puis 
        déclenche le cycle d'apprentissage standardisé.

        Args:
            env (MultiAgentEnv): L'environnement de simulation cible.Ses propriétés
                fondamentales (`env_name`, `map_name`) ainsi que le dictionnaire de paramètres
                dynamiques (`env_kwargs`) en sont extraits pour être injectés dans l'algorithme.
            algorithme (Union[type, Algo]): Classe ou instance de l'algorithme (ex: PPOAlgo).
            architecture (ArchitectureSupport, optional): Configuration réseau si l'algo 
                n'est pas encore instancié.
            algo_hpo_space (AlgoHyperparametersResearchSpace, optional): Espace de recherche 
                pour les paramètres de l'algorithme.
            archi_hpo_space (ArchiHyperparametersResearchSpace, optional): Espace de recherche 
                pour les paramètres de l'architecture.
            stop_criteria (dict, optional): Conditions d'arrêt (ex: {"training_iteration": 3}).
            GPUs (int): Nombre de GPUs à allouer.
            Checkpoints_freq (int): Fréquence de sauvegarde des modèles.
        
        Returns:
            JointPolicy: Politique entraînée prête à l'emploi.
        """
        
        print(f"--- [MARIO ENGINE] Démarrage de la session ---")
        
        # Extraction des identifiants via le wrapper d'environnement
        # Note : On s'appuie sur les attributs spécifiques au wrapper (ex: PettingZooEnvWrapper)
        env_name = env.env_name
        env_kwargs = env.env_kwargs
        map_name = env.map_name
        
        

        # Exectution avec optimisation d'hyperparamètres (Optuna)
        if algo_hpo_space is not None and archi_hpo_space is not None:
            print("[MARIO ENGINE] Mode HPO activé (Optuna)")
            from mario.hpo.optimizer import HPOptimizer

            optimizer = HPOptimizer(
                algo_class=algorithme,
                algo_space=algo_hpo_space,
                archi_space=archi_hpo_space,
                env_name=env_name,
                map_name=map_name,
                env_kwargs=env_kwargs,
                n_trials=n_trials,
                n_workers=n_workers,
                training_iterations=hpo_training_iterations,
                direction=hpo_direction,
                stop_criteria=stop_criteria,
                GPUs=GPUs,
                Checkpoints_freq=Checkpoints_freq,
            )
            policy, study = optimizer.optimize()

        # Execution sans optimiasation d'hyperparamètres
        else:
            print("[MARIO ENGINE] Mode entraînement standard")

            algo_instance = algorithme(architecture, algo_hpo_space)

            policy = algo_instance.train(
                env_name=env_name,
                map_name=map_name,
                env_kwargs=env_kwargs,
                architecture=architecture,
                stop_criteria=stop_criteria,
                GPUs=GPUs,
                Checkpoints_freq=Checkpoints_freq,
            )

        print(f"--- [MARIO ENGINE] Entrainement terminé ! ---")
        return policy

    def run_render(self,policy,env,save_mode="human"):
        """
        Exécute la visualisation d'une politique entraînée dans son environnement.

        Cette méthode délègue l'affichage ou la génération de médias à l'objet
        `policy`. Elle permet de voir les agents en action, soit en temps réel,
        soit via l'exportation de fichiers vidéo/GIF.

        Args:
            policy (JointPolicy): La politique entraînée à visualiser (généralement
                une instance de `MARLlibPolicy`). Elle doit posséder une méthode `.render()`.
            env (MultiAgentEnv): L'environnement de simulation. Bien que certains
                moteurs (comme MARLlib) recréent l'environnement en interne, il est
                passé ici pour maintenir la cohérence avec l'interface MARIO.
            save_mode (str, optional): Définit le mode de sortie de la visualisation.

                - `"human"` (défaut) : Ouvre une fenêtre graphique pour un rendu en direct.
                - `"video"` : Enregistre une vidéo au format `.mp4`.
                - `"gif"` : Enregistre une animation au format `.gif`.

        Returns:
            bool: `True` si le processus de rendu s'est terminé avec succès.

        Example:
            ```python
            engine = RunEngine()
            # Visualisation directe
            engine.run_render(trained_policy, my_env, save_mode="human")

            # Exportation en vidéo
            engine.run_render(trained_policy, my_env, save_mode="video")
            ```

        Note:
            Pour les modes `"video"` et `"gif"`, les fichiers sont généralement
            sauvegardés dans le dossier `renders/` à l'intérieur du répertoire
            de l'expérience correspondante.
        """
        policy.render(env=env, model=policy.model, save_mode=save_mode)
        return True