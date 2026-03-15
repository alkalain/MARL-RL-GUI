import time
from pettingzoo.butterfly import pistonball_v6

from mario.engine import RunEngine
from mario.envs.pettingzoo_env import PettingZooEnvWrapper
from mario.algos.ppo import PPOAlgo
from mario.algos.base import ArchitectureSupport
from mario.hpo.spaces import AlgoHyperparametersResearchSpace, ArchiHyperparametersResearchSpace

# --- Bouchons (dummies) pour les modules non encore implémentés ---
class DummyArch(ArchitectureSupport):
    def __init__(self): super().__init__("Dummy_CNN")

class DummyAlgoSpace(AlgoHyperparametersResearchSpace):
    def __init__(self): super().__init__("Dummy_Algo_Space")

class DummyArchiSpace(ArchiHyperparametersResearchSpace):
    def __init__(self): super().__init__("Dummy_Archi_Space")

# 1. Chargement de l'environnement PettingZoo avec notre Wrapper MARIO
raw_env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
mario_env = PettingZooEnvWrapper(raw_env)

# 2. Préparation de l'algorithme PPO et du moteur
mario_algo = PPOAlgo()
engine = RunEngine()

# 3. Exécution de la simulation d'entraînement
policy = engine.run_training(
    env=mario_env,
    algo=mario_algo,
    architecture=DummyArch(),
    algo_hpo_space=DummyAlgoSpace(),
    archi_hpo_space=DummyArchiSpace()
)

# 4. Lancement visuel du jeu
observations, infos = mario_env.reset()

for step in range(150):
    # Génération d'actions aléatoires (en attendant la vraie fonction predict() de la policy)
    actions = {agent: raw_env.action_space(agent).sample() for agent in raw_env.agents}
    
    # Avancement du jeu via le Wrapper MARIO
    obs, rewards, terminations, truncations, infos = mario_env.step(actions)
    time.sleep(0.02)
    
    if not raw_env.agents:
        break

mario_env.close()
