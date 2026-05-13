from abc import ABC, abstractmethod

class JointPolicy(ABC):
    """
    Interface abstraite représentant la politique jointe des agents.
    
    Cette classe sert de base pour encapsuler les modèles entraînés. Elle définit 
    comment une entité de décision doit transformer les perceptions globales 
    en actions individuelles pour chaque agent.
    """
    def __init__(self, policy_type: str):
        """
        Initialise le type de politique.

        Args:
            policy_type (str): Étiquette identifiant la nature de la politique 
                (ex: "MARLlib_Policy", "Scripted_Policy").
        """
        self.type = policy_type

    @abstractmethod
    def predict(self, observations):
        """
        Détermine les actions à exécuter en fonction des observations actuelles.

        Args:
            observations: État perçu de l'environnement (format dépendant du moteur utilisé).

        Returns:
            dict: Un dictionnaire associant l'identifiant de chaque agent à son action 
                correspondante (ex: {'agent_0': 1, 'agent_1': 0}).
        """
        pass

class Algo(ABC):
    """
    Interface de base pour la configuration et la gestion des algorithmes.
    
    Cette classe définit le cadre nécessaire pour implémenter des algorithmes 
    d'apprentissage (ex: PPO, MAPPO) de manière indépendante des autres modules.
    """
    def __init__(self, algo_type: str):
        """
        Initialise les propriétés fondamentales de l'algorithme.

        Args:
            algo_type (str): Nom explicite de l'algorithme utilisé.
        """
        self.type = algo_type

class ArchitectureSupport(ABC):
    """
    Interface dédiée à la structure des réseaux de neurones.
    
    Elle permet de définir la configuration structurelle (couches, types de neurones) 
    qui sera injectée dans l'algorithme pour construire le modèle d'apprentissage.
    """
    def __init__(self, arch_type: str):
        """
        Initialise le type d'architecture.

        Args:
            arch_type (str): Identifiant de l'architecture (ex: "MLP", "CNN", "RNN").
        """
        self.type = arch_type

    @abstractmethod
    def to_marllib_config(self) -> dict:
        """
        Convertit les paramètres de l'objet en un format compatible avec MARLlib.

        Returns:
            dict: Configuration prête à être transmises aux fonctions de 
                construction de modèle de MARLlib.
        """
        pass