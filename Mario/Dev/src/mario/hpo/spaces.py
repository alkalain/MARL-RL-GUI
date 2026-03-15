from abc import ABC

class AlgoHyperparametersResearchSpace(ABC):
    """Définit les limites de recherche (min/max) pour les paramètres de l'algorithme."""
    def __init__(self, space_type: str):
        self.type = space_type

class ArchiHyperparametersResearchSpace(ABC):
    """Définit les limites de recherche (min/max) pour l'architecture du réseau de neurones."""
    def __init__(self, space_type: str):
        self.type = space_type