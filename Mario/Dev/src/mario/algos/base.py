from abc import ABC, abstractmethod
from marllib import marl
from mario.algos.marllibpolicy import MARLlibPolicy

class ArchitectureSupport(ABC):
    """
    Interface dédiée à la structure des réseaux de neurones.
    
    Elle permet de définir la configuration structurelle (couches, types de neurones) 
    qui sera injectée dans l'algorithme pour construire le modèle d'apprentissage.
    """
    def __init__(self, arch_type: str):
        """
        Initialise le type d'architecture.

        Args:
            arch_type (str): Identifiant de l'architecture (ex: "MLP", "CNN", "RNN").
        """
        self.type = arch_type

    @abstractmethod
    def to_marllib_config(self) -> dict:
        """
        Convertit les paramètres de l'objet en un format compatible avec MARLlib.

        Returns:
            dict: Configuration prête à être transmises aux fonctions de 
                construction de modèle de MARLlib.
        """
        pass


class Algo(ABC):
    """
    Interface de base pour la configuration et l'entraînement des algorithmes.

    Cette classe fournit le moteur d'exécution standardisé (`train`) pour 
    l'ensemble des algorithmes MARIO. Elle permet d'abstraire la complexité 
    des appels MARLlib tout en garantissant une interface uniforme pour 
    l'utilisateur final.
    """
    def __init__(self, algo_type: str, hyperparams: dict = None):
        """
        Initialise les propriétés fondamentales de l'algorithme.

        Args:
            algo_type (str): Nom explicite de l'algorithme utilisé.
            hyperparams (dict, optional): Paramètres spécifiques à l'optimisation.
        """
        self.type = algo_type
        self.hyperparams = hyperparams or {}

    @abstractmethod
    def _get_marllib_algo(self, env_name: str):
        """
        Retourne l'instance d'algorithme configurée pour MARLlib.

        Args:
            env_name (str): Nom de l'environnement pour la source des hyperparamètres.

        Returns:
            Une instance d'algorithme MARLlib (ex: mappo, qmix).
        """
        pass

    def train(self, env_name: str, map_name: str, architecture, 
              stop_criteria: dict = None, GPUs: int = 0, Checkpoints_freq: int = 1) -> MARLlibPolicy:
        """
        Exécute le cycle d'apprentissage multi-agent.

        Cette méthode orchestre :
        1. L'instanciation de l'environnement.
        2. La récupération de la configuration algorithmique via `_get_marllib_algo`.
        3. La construction du modèle selon l'architecture fournie.
        4. Le lancement de l'entraînement (`fit`).

        Args:
            env_name (str): Identifiant de l'environnement (ex: 'mpe').
            map_name (str): Scénario spécifique.
            architecture: Instance d'ArchitectureSupport.
            stop_criteria (dict): Conditions d'arrêt (défaut : 10 itérations).
            GPUs (int): Allocation de ressources GPU.
            Checkpoints_freq (int): Fréquence de sauvegarde.

        Returns:
            MARLlibPolicy: Interface de décision entraînée.
        """
        print(f"[MARIO] Initialisation {env_name}:{map_name} | Algo: {self.type} | Archi: {architecture.type}")
        
        env_output = marl.make_env(environment_name=env_name, map_name=map_name)
        
        # Appel polymorphe pour obtenir l'algo spécifique
        algo_instance = self._get_marllib_algo(env_name)
        
        arch_config = architecture.to_marllib_config()
        model = marl.build_model(env_output, algo_instance, arch_config)

        print(f"[MARIO] Début entraînement | Hyperparams : {self.hyperparams}")
        algo_instance.fit(
            env_output, model,
            stop=stop_criteria,
            local_mode=True,
            num_gpus=GPUs,
            checkpoint_freq=Checkpoints_freq,
            **self.hyperparams
        )
        
        # Extraction de l'env pour la politique
        env = env_output[0] if isinstance(env_output, tuple) else env_output
        exp_pattern = f"{self.type.split(' ')[0].lower()}_{architecture.type.lower()}_{map_name}/MAPPOTrainer_*"
        
        return MARLlibPolicy(model, algo_instance, env, exp_pattern)