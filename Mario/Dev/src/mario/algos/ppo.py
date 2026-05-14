import os
import glob
from pathlib import Path
from marllib import marl
from mario.algos.base import Algo, ArchitectureSupport
from mario.algos.architectures import MLPArchitecture
from mario.algos.marllibpolicy import MARLlibPolicy

class PPOAlgo(Algo):
    """
    Implémentation de l'algorithme Proximal Policy Optimization (PPO) 
    s'appuyant sur le framework MARLlib.

    Cette classe gère la configuration des hyperparamètres d'apprentissage et 
    l'orchestration du cycle de vie de l'entraînement, de l'initialisation de 
    l'environnement à la génération de la politique finale.
    """
    def __init__(
            self,
            architecture: ArchitectureSupport = None,
            hyperparams: dict = None,
            ):
        """
        Configure l'algorithme avec une architecture et des paramètres d'optimisation.

        Args:
            architecture (ArchitectureSupport): Définition du réseau de neurones. 
                Par défaut : MLPArchitecture (Perceptron Multicouche).
            hyperparams (dict): Dictionnaire de configuration de l'apprentissage. 
                Valeurs par défaut : lr=0.0005, batch_size=512, sgd_iter=10.
        """
        super().__init__(algo_type="PPO (MARLlib)")
        self.architecture = architecture or MLPArchitecture()

        self.hyperparams = hyperparams or {
            "lr": 0.0005,
            "train_batch_size": 512,
            "num_sgd_iter": 10,
        }


    def train(self, env_name: str, map_name: str, stop_criteria: dict = None, GPUs=0, Checkpoints_freq=1) -> MARLlibPolicy:
        """
        Exécute le processus complet d'apprentissage par renforcement multi-agent.

        La méthode suit quatre étapes :
        1. Instanciation de l'environnement via l'interface MARLlib.
        2. Configuration de l'algorithme MA-PPO (Multi-Agent PPO).
        3. Construction du modèle neuronal selon l'architecture spécifiée.
        4. Lancement de l'optimisation (méthode fit).

        Args:
            env_name (str): Identifiant de l'environnement (ex: 'pettingzoo').
            map_name (str): Scénario spécifique au sein de l'environnement.
            stop_criteria (dict): Conditions d'arrêt de l'entraînement. 
                Défaut : 10 itérations.
            GPUs (int): Nombre de ressources graphiques allouées.
            Checkpoints_freq (int): Fréquence de sauvegarde de l'état du modèle.

        Returns:
            MARLlibPolicy: Une instance prête à l'emploi contenant le modèle entraîné.
        """
        if stop_criteria is None:
            stop_criteria = {"training_iteration": 10}
        print(f"[MARIO] Initialisation de l'environnement {env_name}:{map_name}| Archi: {self.architecture.type}")
        # 1. Configuration de l'environnement
        env = marl.make_env(environment_name=env_name, map_name=map_name)

        # 2. Configuration de l'algorithme (MA-PPO est l'implémentation standard)
        mappo = marl.algos.mappo(hyperparam_source=env_name)

        # 3. Construction du modèle avec l'architecture définie
        arch_config = self.architecture.to_marllib_config()
        print(f"[MARIO] Config architecture : {arch_config}")
        model = marl.build_model(env, mappo, arch_config)

        # 4. Lancement de l'entraînement
        print(f"[MARIO] Début entraînement | Hyperparams algo : {self.hyperparams}")
        mappo.fit(
            env, model,
            stop=stop_criteria,
            local_mode=True,
            num_gpus=GPUs,
            checkpoint_freq=Checkpoints_freq,
            **self.hyperparams
            )

        exp_pattern = f"mappo_{self.architecture.type.lower()}_{map_name}/MAPPOTrainer_*"

        return MARLlibPolicy(model, mappo, env, exp_pattern)