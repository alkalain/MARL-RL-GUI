from marllib import marl
from mario.algos.base import Algo, JointPolicy

class MARLlibPolicy(JointPolicy):
    """
    Wrapper pour transformer le résultat de MARLlib en une 
    politique compatible avec l'interface MARIO.
    """
    def __init__(self, model, algo_instance, env_instance):
        super().__init__(policy_type="MARLlib_Policy")
        self.model = model
        self.algo = algo_instance
        self.env = env_instance

    def predict(self, observations):
        """
        À implémenter plus tard : utilisera self.algo.render ou 
        self.model pour calculer les actions.
        """
        return {}

class PPOAlgo(Algo):
    """Configuration et logique d'entraînement via MARLlib."""
    def __init__(self, hyperparams: dict = None):
        super().__init__(algo_type="PPO (MARLlib)")
        # Paramètres par défaut si rien n'est fourni
        self.hyperparams = hyperparams if hyperparams else {
            "core_arch": "mlp",
            "encode_layer": "128-128"
        }

    def train(self, env_name: str, map_name: str, stop_criteria: dict = None) -> MARLlibPolicy:
        """
        Lance la boucle d'apprentissage MARLlib.
        """
        if stop_criteria is None:
            stop_criteria = {"training_iteration": 10}

        print(f"[MARIO] Initialisation de l'environnement {env_name}:{map_name}...")
        # 1. Configuration de l'environnement
        env = marl.make_env(environment_name=env_name, map_name=map_name)

        # 2. Configuration de l'algorithme (MA-PPO est l'implémentation standard)
        mappo = marl.algos.mappo(hyperparam_source=env_name)

        # 3. Construction du modèle avec l'architecture définie
        model = marl.build_model(env, mappo, self.hyperparams)

        print(f"[MARIO] Début de l'entraînement MARLlib...")
        # 4. Lancement de l'entraînement
        mappo.fit(env, model,
                  stop=stop_criteria,
                  local_mode=True,
                  num_gpus=0,
                  checkpoint_freq=1)

        return MARLlibPolicy(model, mappo, env)