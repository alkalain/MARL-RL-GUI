from mario.algos.base import ArchitectureSupport

class MLPArchitecture(ArchitectureSupport):
    """
    Class convertissant une architecture de réseau de neurones MARLlib en un objet MARIO.
    
    Cette classe implémente un réseau de type perceptron multicouche (MLP), 
    particulièrement adapté au traitement d'observations vectorielles.

    Chaque classe héritant de ce support doit :
    1. Définir une configuration par défaut
    2. Exposer les hyperparamètres requis par MARLlib pour la construction du modèle.
    """
    def __init__(self, layers: str = "128-128"):
        """
        Initialise l'architecture avec une structure de couches définie.

        Args:
            layers (str): Configuration des couches cachées
            Exemple : "256-128-64" = réseau à trois couches cachées.
        """
        super().__init__(arch_type="MLP")
        self.layers = layers

    def to_marllib_config(self) -> dict:
        """
        Génère le dictionnaire de configuration conforme aux spécifications de MARLlib.

        Returns:
            dict: Paramètres d'architecture incluant le type de coeur et les couches d'encodage.
            Exemple : {'core_arch': 'mlp', 'encode_layer': '128-128'}
        """
        return {
            "core_arch": "mlp",
            "encode_layer": self.layers
        }

class GRUArchitecture(ArchitectureSupport):
    """Réseau récurrent GRU — adapté aux environnements partiellement observables."""
    def __init__(self, layers: str = "128-128", hidden_state_size: int = 256):
        super().__init__(arch_type="GRU")
        self.layers = layers
        self.hidden_state_size = hidden_state_size

    def to_marllib_config(self) -> dict:
        return {
            "core_arch": "gru",
            "encode_layer": self.layers,
            "hidden_state_size": self.hidden_state_size,
        }

class CNNArchitecture(ArchitectureSupport):
    """Réseau convolutif — adapté aux observations image."""
    def __init__(self, layers: str = "64-64"):
        super().__init__(arch_type="CNN")
        self.layers = layers

    def to_marllib_config(self) -> dict:
        return {
            "core_arch": "cnn",
            "encode_layer": self.layers,
        }

class RNNArchitecture(ArchitectureSupport):
    """Réseau récurrent — adapté aux séquences temporelles."""
    def __init__(self, layers: str = "64-64"):
        super().__init__(arch_type="RNN")
        self.layers = layers

    def to_marllib_config(self) -> dict:
        return {
            "core_arch": "rnn",
            "encode_layer": self.layers,
        }