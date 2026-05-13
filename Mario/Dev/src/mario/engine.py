from mario.envs.base import MultiAgentEnv
from mario.algos.base import Algo, ArchitectureSupport, JointPolicy
from mario.hpo.spaces import AlgoHyperparametersResearchSpace, ArchiHyperparametersResearchSpace
from mario.utils.stats import Stats

class RunEngine:
    """
    Moteur d'exécution principal chargé de coordonner le cycle de vie des processus.
    
    Cette classe centralise la logique de pilotage en orchestrant les interactions 
    entre l'environnement, l'algorithme d'apprentissage et les structures de recherche 
    d'hyperparamètres.
    """
    def __init__(self, engine_type: str = "default"):
        """
        Initialise l'instance du moteur d'exécution.

        Args:
            engine_type (str): Identifiant du moteur (défaut : "default").
        """
        self.type = engine_type

    def run_training(
        self,
        env: MultiAgentEnv,
        algo: Algo,
        architecture: ArchitectureSupport = None,
        algo_hpo_space: AlgoHyperparametersResearchSpace = None,
        archi_hpo_space: ArchiHyperparametersResearchSpace = None,
        stop_criteria: dict = None,
        GPUs=0,
        Checkpoints_freq=1
    ) -> JointPolicy:
        """
        Pilote une session complète d'entraînement automatique.
        
        Cette méthode extrait les configurations nécessaires des composants fournis 
        et lance la procédure d'apprentissage pour générer une politique de décision.

        Args:
            env (MultiAgentEnv): L'environnement de simulation cible.
            algo (Algo): L'algorithme d'apprentissage à utiliser.
            architecture (ArchitectureSupport, optional): La configuration du réseau de neurones.
            algo_hpo_space (AlgoHyperparametersResearchSpace, optional): Espace de recherche 
                pour les paramètres de l'algorithme.
            archi_hpo_space (ArchiHyperparametersResearchSpace, optional): Espace de recherche 
                pour les paramètres de l'architecture.
            stop_criteria (dict, optional): Conditions d'arrêt de l'entraînement.
        
        Returns:
            JointPolicy: La politique jointe résultante de l'entraînement, prête pour l'exécution.
        """
        
        print(f"--- [MARIO ENGINE] Démarrage de la session ---")
        
        # Extraction des identifiants via le wrapper d'environnement
        # Note : On s'appuie sur les attributs spécifiques au wrapper (ex: PettingZooEnvWrapper)
        env_name = env.env_name 
        map_name = env.map_name

        algorithme = algo(architecture, algo_hpo_space)

        # On lance l'entraînement
        # On adapte l'appel pour que l'algo reçoive ce dont il a besoin
        policy = algo.train(
            env_name=env_name,
            map_name=map_name,
            stop_criteria=stop_criteria,
            GPUs=GPUs,
            Checkpoints_freq=Checkpoints_freq,
        )

        print(f"--- [MARIO ENGINE] Entrainement terminé ! ---")
        return policy

    def run_render(self,policy,env,save_mode="human"):
        policy.render(env=env, model=policy.model, save_mode=save_mode)
        return True