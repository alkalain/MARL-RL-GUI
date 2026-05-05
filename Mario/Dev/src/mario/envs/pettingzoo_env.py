from mario.envs.base import ParallelEnv

class PettingZooEnvWrapper(ParallelEnv):
    """Adaptateur pour rendre un jeu PettingZoo compatible avec le moteur MARIO."""
    def __init__(self, env_name: str, map_name: str):
        # On passe le type à la classe mère (ParallelEnv)
        super().__init__(env_type="PettingZoo")
        
        # Ces deux variables sont les clés pour MARLlib
        self.env_name = env_name 
        self.map_name = map_name 
        
        # Pour l'instant, on laisse l'instance interne à None.
        # MARLlib créera sa propre instance en interne via marl.make_env()
        self.env = None

    def reset(self):
        """Réinitialise le jeu (utilisé uniquement hors MARLlib)."""
        if self.env:
            return self.env.reset()
        return None

    def step(self, actions):
        """Applique les actions (utilisé uniquement hors MARLlib)."""
        if self.env:
            return self.env.step(actions)
        return None
        
    def close(self):
        """Ferme l'environnement."""
        if self.env:
            self.env.close()