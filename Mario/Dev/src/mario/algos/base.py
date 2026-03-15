from abc import ABC, abstractmethod

class JointPolicy(ABC):
    """Interface représentant le 'cerveau' entraîné des agents."""
    def __init__(self, policy_type: str):
        self.type = policy_type

    @abstractmethod
    def predict(self, observations):
        """Déduit les actions à prendre en analysant les observations de l'environnement."""
        pass

class Algo(ABC):
    """Interface de base pour la configuration d'un algorithme (ex: PPO, MAPPO)."""
    def __init__(self, algo_type: str):
        self.type = algo_type

class ArchitectureSupport(ABC):
    """Interface pour la structure du réseau de neurones (ex: CNN, MLP)."""
    def __init__(self, arch_type: str):
        self.type = arch_type