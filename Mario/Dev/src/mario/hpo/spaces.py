from abc import ABC, abstractmethod
import optuna

class AlgoHyperparametersResearchSpace(ABC):
    """
    Interface définissant l'espace de recherche pour les paramètres de l'algorithme.
    
    Cette classe permet de spécifier les bornes (min/max) et les types de valeurs 
    possibles pour l'optimisation des hyperparamètres de l'apprentissage (ex: learning rate).
    """
    def __init__(self, space_type: str):
        """
        Initialise l'espace de recherche algorithmique.

        Args:
            space_type (str): Identifiant du type d'espace défini.
        """
        self.type = space_type

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> dict:
        """Retourne un dict d'hyperparamètres suggérés par Optuna."""
        pass

class ArchiHyperparametersResearchSpace(ABC):
    """
    Interface définissant l'espace de recherche pour la structure du réseau.
    
    Cette classe permet de définir les variations possibles de l'architecture 
    du réseau de neurones (ex: nombre de couches, nombre de neurones) lors 
    d'une phase d'optimisation.
    """
    def __init__(self, space_type: str):
        """
        Initialise l'espace de recherche architectural.

        Args:
            space_type (str): Identifiant du type d'espace défini.
        """
        self.type = space_type

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> dict:
        """Retourne une config d'architecture suggérée par Optuna."""
        pass

class PPOAlgoSpace(AlgoHyperparametersResearchSpace):
    """Espace de recherche des hyperparamètres PPO/MAPPO."""
    def __init__(self):
        super().__init__("PPO_Algo_Space")

    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "train_batch_size": trial.suggest_categorical(
                "train_batch_size", [256, 512, 1024]
            ),
            "num_sgd_iter": trial.suggest_int("num_sgd_iter", 5, 20),
        }


class MLPArchiSpace(ArchiHyperparametersResearchSpace):
    """Espace de recherche pour les architectures MLP."""
    def __init__(self):
        super().__init__("MLP_Archi_Space")

    def suggest(self, trial: optuna.Trial) -> dict:
        width = trial.suggest_categorical("layer_width", [64, 128, 256])
        depth = trial.suggest_int("num_layers", 1, 3)
        encode_layer = "-".join([str(width)] * depth)
        return {
            "core_arch": "mlp",
            "encode_layer": encode_layer,
        }
