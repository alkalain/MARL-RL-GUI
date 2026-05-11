from abc import ABC

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