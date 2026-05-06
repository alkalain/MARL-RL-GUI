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
        architecture: ArchitectureSupport = None,
        algo_hpo_space: AlgoHyperparametersResearchSpace = None,
        archi_hpo_space: ArchiHyperparametersResearchSpace = None
    ) -> JointPolicy:
        
        print(f"--- [MARIO ENGINE] Démarrage de la session ---")
        
        # Pour MARLlib, on va extraire les infos du wrapper env
        # au lieu de faire env.reset() nous-mêmes
        env_name = env.env_name # (là on part du principe que le wrapper a cet attribut là)
        map_name = env.map_name
        
        # On lance l'entraînement
        # On adapte l'appel pour que l'algo reçoive ce dont il a besoin
        policy = algo.train(
            env_name=env_name, 
            map_name=map_name, 
            stop_criteria={"training_iteration": 10}
        )
        
        print(f"--- [MARIO ENGINE] Entrainement terminé ! ---")
        return policy