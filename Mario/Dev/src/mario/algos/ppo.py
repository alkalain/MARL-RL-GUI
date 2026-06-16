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
        super().__init__(
            algo_type="mappo",
            hyperparams=hyperparams or {
                "lr": 0.0005,
                "train_batch_size": 512,
                "num_sgd_iter": 10,
            },
            share_policy="group"
        )
        self.architecture = architecture or MLPArchitecture()


    def _get_marllib_algo(self, env_name: str):
        """
        Expose l'implémentation MA-PPO de MARLlib.

        Args:
            env_name (str): Nom de l'environnement pour l'initialisation.

        Returns:
            L'instance de l'algorithme MA-PPO.
        """
        return marl.algos.mappo(hyperparam_source=env_name)
