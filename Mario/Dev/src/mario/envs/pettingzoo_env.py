from mario.envs.base import ParallelEnv

class PettingZooEnvWrapper(ParallelEnv):
    """Adaptateur pour rendre un jeu PettingZoo compatible avec le moteur MARIO."""
    def __init__(self, pz_env):
        super().__init__(env_type="PettingZoo_Parallel")
        self.env = pz_env 

    def reset(self):
        """Réinitialise le jeu PettingZoo."""
        observations, infos = self.env.reset()
        return observations, infos

    def step(self, actions):
        """Applique les actions des agents dans le jeu PettingZoo."""
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, infos
        
    def close(self):
        """Ferme la fenêtre visuelle de l'environnement."""
        self.env.close()