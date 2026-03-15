from mario.envs.base import MultiAgentEnv
from mario.algos.base import Algo, ArchitectureSupport, JointPolicy
from mario.hpo.spaces import AlgoHyperparametersResearchSpace, ArchiHyperparametersResearchSpace
from mario.utils.stats import Stats

class RunEngine:
    def __init__(self, engine_type: str = "default"):
        self.type = engine_type

    def run_training(
        self,
        env: MultiAgentEnv,
        algo: Algo,
        architecture: ArchitectureSupport,
        algo_hpo_space: AlgoHyperparametersResearchSpace,
        archi_hpo_space: ArchiHyperparametersResearchSpace
    ) -> JointPolicy:
        
        print(f"--- [MARIO ENGINE] Démarrage de la session ---")
        
        # 1. On initialise l'environnement
        env.reset()
        
        # 2. (Plus tard, Optuna interviendra ici avec les algo_hpo_space)
        print(f"[MARIO ENGINE] Bypass Optuna pour l'instant (Architecture: {architecture.type})")
        
        # 3. On lance l'entrainement via notre wrapper d'algorithme
        # On suppose que l'algo possède une méthode .train() (comme notre PPOAlgo)
        policy = algo.train(env=env, total_timesteps=1000)
        
        print(f"--- [MARIO ENGINE] Entrainement terminé ! ---")
        return policy

    def run_test(self, policy: JointPolicy, env: MultiAgentEnv) -> Stats:
        pass