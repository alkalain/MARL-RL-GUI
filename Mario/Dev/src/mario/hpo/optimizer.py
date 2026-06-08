import optuna
from typing import Type, Optional
from mario.hpo.spaces import AlgoHyperparametersResearchSpace, ArchiHyperparametersResearchSpace
from mario.algos.base import Algo, JointPolicy


class HPOptimizer:
    """
    Optimiseur Optuna générique pour tout algorithme MARIO + MARLlib.

    Compatible avec tout algorithme héritant de `Algo` et tout espace de
    recherche héritant de `AlgoHyperparametersResearchSpace` /
    `ArchiHyperparametersResearchSpace`.
    """

    def __init__(
        self,
        algo_class: Type[Algo],
        algo_space: AlgoHyperparametersResearchSpace,
        archi_space: ArchiHyperparametersResearchSpace,
        env_name: str,
        map_name: str,
        env_kwargs: dict = None,
        n_trials: int = 10,
        training_iterations: int = 5,
        direction: str = "maximize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_name: Optional[str] = None,
        stop_criteria: dict = None,
        GPUs: int = 0,
        Checkpoints_freq: int = 1,
    ):
        self.algo_class = algo_class
        self.algo_space = algo_space
        self.archi_space = archi_space
        self.env_name = env_name
        self.map_name = map_name
        self.env_kwargs = env_kwargs or {}
        self.n_trials = n_trials
        self.training_iterations = training_iterations
        self.direction = direction
        self.pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=3)
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.study_name = study_name or f"mario_hpo_{algo_class.__name__}_{map_name}"
        self.stop_criteria = stop_criteria or {"training_iteration": training_iterations}
        self.GPUs = GPUs
        self.Checkpoints_freq = Checkpoints_freq
        self.best_policy: Optional[JointPolicy] = None
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Fonction objectif appelée par Optuna à chaque essai."""
        algo_params = self.algo_space.suggest(trial)
        archi_params = self.archi_space.suggest(trial)

        print(f"\n[MARIO HPO] Trial #{trial.number}")
        print(f"  Algo params  : {algo_params}")
        print(f"  Archi params : {archi_params}")

        from mario.algos.architectures import MLPArchitecture, GRUArchitecture, CNNArchitecture
        arch_map = {"mlp": MLPArchitecture, "gru": GRUArchitecture, "cnn": CNNArchitecture}
        arch_class = arch_map.get(archi_params.get("core_arch", "mlp"), MLPArchitecture)
        architecture = arch_class(layers=archi_params.get("encode_layer", "128-128"))

        algo = self.algo_class(architecture=architecture, hyperparams=algo_params)
        policy = algo.train(
            env_name=self.env_name,
            map_name=self.map_name,
            env_kwargs=self.env_kwargs,
            stop_criteria=self.stop_criteria,
            GPUs=self.GPUs,
            Checkpoints_freq=self.Checkpoints_freq,
        )

        score = self._evaluate(policy, trial)

        # Garde la meilleure politique
        if self.best_policy is None or \
           (self.direction == "maximize" and score >= self.study.best_value) or \
           (self.direction == "minimize" and score <= self.study.best_value):
            self.best_policy = policy

        return score

    def _evaluate(self, policy: JointPolicy, trial: optuna.Trial) -> float:
        """Lit le score depuis result.json du run le plus récent."""
        import json, glob, os
        from pathlib import Path

        src_dir = Path(__file__).resolve().parent.parent.parent
        pattern = str(
            src_dir / "exp_results"
            / f"mappo_*_{self.map_name}"
            / "MAPPOTrainer_*"
            / "result.json"
        )
        result_files = glob.glob(pattern)

        if not result_files:
            print("[MARIO HPO] Aucun result.json trouvé, score = 0.0")
            return 0.0

        latest = max(result_files, key=os.path.getctime)
        try:
            with open(latest, "r") as f:
                lines = [l for l in f.readlines() if l.strip()]
                last = json.loads(lines[-1])
                score = last.get("episode_reward_mean", 0.0)
                print(f"[MARIO HPO] Score trial #{trial.number} : {score:.4f}")
                trial.report(score, step=self.training_iterations)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                return score
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"[MARIO HPO] Erreur lecture score : {e}")
            return 0.0

    def optimize(self) -> tuple:
        """Lance l'optimisation Optuna et retourne (best_policy, study)."""
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
        )

        print(f"\n[MARIO HPO] Démarrage : {self.study_name}")
        print(f"  Essais     : {self.n_trials}")
        print(f"  Direction  : {self.direction}")
        print(f"  Iterations : {self.training_iterations} par essai\n")

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            catch=(Exception,),
        )

        print("\n[MARIO HPO] Optimisation terminée !")

        completed = [t for t in self.study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed:
            print("[MARIO HPO] Aucun essai complété — vérifier les erreurs ci-dessus.")
            return self.best_policy, self.study


        print(f"  Meilleur trial  : #{self.study.best_trial.number}")
        print(f"  Meilleur score  : {self.study.best_value:.4f}")
        print(f"  Meilleurs params: {self.study.best_params}")

        return self.best_policy, self.study