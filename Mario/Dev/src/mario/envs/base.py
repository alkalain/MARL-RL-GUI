from abc import ABC, abstractmethod

class MultiAgentEnv(ABC):
    """
    Interface qui définie le fonctionnement des environnements 
    multi-agents au sein de MARIO.
    
    Cette classe sert de modèle pour s'assurer que n'importe quel moteur 
    de jeu (comme PettingZoo) puisse communiquer correctement avec 
    nos algorithmes d'apprentissage.
    """
    def __init__(self, env_type: str):
        """
        Initialise les propriétés de l'environnement.

        Args:
            env_type (str): Catégorie ou moteur de l'environnement 
                (ex: "PettingZoo", "SMAC", "Flatland").
        """
        self.type = env_type

    @abstractmethod
    def reset(self):
        """
        Réinitialise l'état de la simulation à ses conditions initiales.

        Returns:
            observations: Les perceptions initiales de l'ensemble des agents 
                après la remise à zéro.
        """
        pass

    @abstractmethod
    def step(self, actions):
        """
        Fait progresser la simulation d'une étape (time-step).
        
        Cette méthode applique les actions choisies par les agents et calcule 
        la transition vers l'état suivant.

        Args:
            actions (dict): Un dictionnaire associant l'identifiant de chaque 
                agent à l'action qu'il doit exécuter.
        
        Returns:
            tuple: Un ensemble contenant les nouvelles observations, les récompenses, 
                les indicateurs de terminaison et les informations additionnelles.
        """
        pass

class ParallelEnv(MultiAgentEnv):
    """
    Spécialisation de l'interface pour les environnements à exécution synchrone.
    
    Cette classe est dédiée aux simulations où l'ensemble des agents prennent 
    leurs décisions et agissent simultanément à chaque étape du cycle de contrôle.
    """
    def __init__(self, env_type: str):
        """
        Initialise l'environnement parallèle.

        Args:
            env_type (str): Nom du moteur de simulation sous-jacent.
        """
        super().__init__(env_type)