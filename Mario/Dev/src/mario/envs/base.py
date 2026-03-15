from abc import ABC, abstractmethod

class MultiAgentEnv(ABC):
    """Interface de base pour tous les environnements multi-agents."""
    def __init__(self, env_type: str):
        self.type = env_type

    @abstractmethod
    def reset(self):
        """Réinitialise l'environnement au point de départ."""
        pass

    @abstractmethod
    def step(self, actions):
        """Fait avancer l'environnement d'une étape en fonction des actions."""
        pass

class ParallelEnv(MultiAgentEnv):
    """Interface pour les environnements où tous les agents agissent en même temps."""
    def __init__(self, env_type: str):
        super().__init__(env_type)