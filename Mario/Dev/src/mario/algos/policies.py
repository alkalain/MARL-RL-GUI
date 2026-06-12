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