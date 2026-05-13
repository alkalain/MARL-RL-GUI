from mario.envs.base import ParallelEnv

class PettingZooEnvWrapper(ParallelEnv):
    """
    Classe permettant d'intégrer les environnements PettingZoo dans MARIO.
    
    Cette classe fait le lien entre la structure de PettingZoo et les attentes 
    de notre moteur. Elle stocke les identifiants nécessaires pour que MARLlib 
    puisse instancier l'environnement au moment de l'entraînement.
    """
    def __init__(self, env_name: str, map_name: str):
        """
        Initialise le wrapper avec les références de l'environnement.

        Args:
            env_name (str): Nom de l'environnement PettingZoo (ex: 'mpe').
            map_name (str): Nom du scénario spécifique (ex: 'simple_adversary_v3').
        """
        super().__init__(env_type="PettingZoo")
        
        self.env_name = env_name 
        self.map_name = map_name 
        
        # Instance interne de l'environnement, initialisée à None par défaut.
        # Note : MARLlib gère généralement sa propre instanciation via marl.make_env().
        self.env = None

    def reset(self):
        """
        Réinitialise l'environnement interne s'il existe.
        
        Note : Cette méthode est principalement destinée à un usage direct, 
        hors du cycle d'entraînement automatisé de MARLlib.
        """
        if self.env:
            return self.env.reset()
        return None

    def step(self, actions):
        """
        Transmet les actions à l'environnement interne et retourne le nouvel état.

        Args:
            actions (dict): Dictionnaire des actions pour chaque agent.
        """
        if self.env:
            return self.env.step(actions)
        return None
        
    def close(self):
        """
        Libère les ressources et ferme l'environnement de simulation.
        """
        if self.env:
            self.env.close()