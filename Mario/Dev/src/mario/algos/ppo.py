from marllib import marl
from mario.algos.base import Algo, ArchitectureSupport
from mario.algos.architectures import MLPArchitecture
from mario.algos.marllibpolicy import MARLlibPolicy

class PPOAlgo(Algo):
    """Configuration et logique d'entraînement via MARLlib."""
    def __init__(
            self,
            architecture: ArchitectureSupport = None,
            hyperparams: dict = None,
            ):
        super().__init__(algo_type="PPO (MARLlib)")
        # Paramètres par défaut si rien n'est fourni
        self.architecture = architecture or MLPArchitecture()

        # Hyperparamètres algo (lr, batch, etc.) séparés de l'archi
        self.hyperparams = hyperparams or {
            "lr": 0.0005,
            "train_batch_size": 512,
            "num_sgd_iter": 10,
        }


    def train(self, env_name: str, map_name: str, stop_criteria: dict = None, GPUs=0, Checkpoints_freq=1) -> MARLlibPolicy:
        """
        Lance la boucle d'apprentissage MARLlib.
        """
        if stop_criteria is None:
            stop_criteria = {"training_iteration": 10}

        print(f"[MARIO] Initialisation de l'environnement {env_name}:{map_name}| Archi: {self.architecture.type}")
        # 1. Configuration de l'environnement
        env = marl.make_env(environment_name=env_name, map_name=map_name)

        # 2. Configuration de l'algorithme (MA-PPO est l'implémentation standard)
        mappo = marl.algos.mappo(hyperparam_source=env_name)

        # 3. Construction du modèle avec l'architecture définie
        arch_config = self.architecture.to_marllib_config()
        print(f"[MARIO] Config architecture : {arch_config}")
        model = marl.build_model(env, mappo, arch_config)

        # 4. Lancement de l'entraînement
        print(f"[MARIO] Début entraînement | Hyperparams algo : {self.hyperparams}")
        mappo.fit(
            env, model,
            stop=stop_criteria,
            local_mode=True,
            num_gpus=GPUs,
            checkpoint_freq=Checkpoints_freq,
            **self.hyperparams
            )

        exp_pattern = f"mappo_{self.architecture.type.lower()}_{map_name}/MAPPOTrainer_*"

        return MARLlibPolicy(model, mappo, env, exp_pattern)