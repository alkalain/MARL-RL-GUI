"""
# `optimizer` — Optimiseur HPO parallèle pour MARIO

Ce module fournit `HPOptimizer`, qui pilote la recherche d'hyperparamètres
via **Optuna** avec parallélisation multi-process sur une seule machine.

## Architecture de parallélisation

Chaque trial Optuna est exécuté dans un **subprocess isolé** (via `joblib/loky`).
Ce choix est contraint par MARLlib : `ray.init()` est appelé en interne à chaque
`algo.train()`, ce qui rend toute tentative de fork du processus principal
non-déterministe (conflit de handles Ray). L'isolation subprocess garantit :

- Un contexte Ray propre par trial (init + shutdown bien délimités).
- Aucune fuite mémoire entre les essais.
- La compatibilité avec `local_mode=True` de MARLlib.

La synchronisation inter-workers est assurée par un **storage SQLite partagé**
(`optuna_storage.db` dans le dossier `exp_results/`). Chaque worker se connecte
à la même étude nommée et les trials sont distribués automatiquement par Optuna.

## Schéma d'exécution

```
optimize()
  └── joblib.Parallel(n_jobs=n_workers)
        ├── _run_worker(trial_ids[0])   → subprocess 1 : ray.init → train → ray.shutdown
        ├── _run_worker(trial_ids[1])   → subprocess 2 : ray.init → train → ray.shutdown
        └── ...
```

## Exemple d'utilisation

```python
from mario.hpo.optimizer import HPOptimizer
from mario.hpo.spaces import PPOAlgoSpace, MLPArchiSpace
from mario.algos.ppo import PPOAlgo

optimizer = HPOptimizer(
    algo_class=PPOAlgo,
    algo_space=PPOAlgoSpace(),
    archi_space=MLPArchiSpace(),
    env_name="mpe",
    map_name="simple_adversary_v3",
    n_trials=12,
    n_workers=3,          # 3 trials en parallèle
    training_iterations=5,
)
best_policy, study = optimizer.optimize()
```

## Dépendances

```bash
pip install optuna joblib
```
"""

import os
import json
import glob
import optuna
import joblib

from pathlib import Path
from typing import Type, Optional
from mario.hpo.spaces import AlgoHyperparametersResearchSpace, ArchiHyperparametersResearchSpace
from mario.algos.base import Algo
from mario.algos.policies import JointPolicy

# Silencer les logs Optuna parasites (INFO de chaque trial)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Fonction top-level exécutée dans chaque subprocess worker
# ---------------------------------------------------------------------------

def _worker_trial(
    study_name: str,
    storage_url: str,
    algo_class_name: str,
    algo_module: str,
    algo_space_class_name: str,
    algo_space_module: str,
    archi_space_class_name: str,
    archi_space_module: str,
    env_name: str,
    map_name: str,
    env_kwargs: dict,
    stop_criteria: dict,
    training_iterations: int,
    GPUs: int,
    Checkpoints_freq: int,
    direction: str,
) -> float:
    """
    Point d'entrée d'un worker subprocess pour un trial Optuna unique.

    Cette fonction est sérialisable (picklable) car elle est définie au niveau
    du module, ce qui est requis par `joblib` avec le backend `loky`.

    Elle reconstruit tous les objets nécessaires depuis leur chemin d'import,
    demande un nouveau trial au storage partagé, exécute l'entraînement complet,
    évalue le score, puis le rapporte à Optuna.

    Args:
        study_name (str): Nom de l'étude Optuna partagée.
        storage_url (str): URL SQLite du storage partagé (ex: ``"sqlite:///path/to/db"``).
        algo_class_name (str): Nom de la classe algorithme à importer.
        algo_module (str): Module Python contenant ``algo_class_name``.
        algo_space_class_name (str): Nom de la classe d'espace algo à importer.
        algo_space_module (str): Module Python contenant ``algo_space_class_name``.
        archi_space_class_name (str): Nom de la classe d'espace architecture à importer.
        archi_space_module (str): Module Python contenant ``archi_space_class_name``.
        env_name (str): Identifiant de l'environnement MARLlib.
        map_name (str): Nom du scénario dans l'environnement.
        env_kwargs (dict): Paramètres dynamiques de l'environnement.
        stop_criteria (dict): Conditions d'arrêt transmises à ``algo.train()``.
        training_iterations (int): Nombre d'itérations d'entraînement par trial.
        GPUs (int): Nombre de GPUs alloués au trial.
        Checkpoints_freq (int): Fréquence de sauvegarde des checkpoints.
        direction (str): ``"maximize"`` ou ``"minimize"``.

    Returns:
        float: Score du trial (``episode_reward_mean`` extrait de ``result.json``),
            ou ``0.0`` en cas d'erreur.
    """
    import importlib

    # --- Reconstruction des objets depuis les noms de classe ---
    algo_mod = importlib.import_module(algo_module)
    algo_class = getattr(algo_mod, algo_class_name)

    algo_space_mod = importlib.import_module(algo_space_module)
    algo_space = getattr(algo_space_mod, algo_space_class_name)()

    archi_space_mod = importlib.import_module(archi_space_module)
    archi_space = getattr(archi_space_mod, archi_space_class_name)()

    # --- Connexion au storage partagé et demande d'un trial ---
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    trial = study.ask()

    try:
        algo_params = algo_space.suggest(trial)
        archi_params = archi_space.suggest(trial)

        worker_id = os.getpid()
        print(f"\n[MARIO HPO][PID {worker_id}] Trial #{trial.number}")
        print(f"  Algo params  : {algo_params}")
        print(f"  Archi params : {archi_params}")

        # --- Construction de l'architecture ---
        from mario.algos.architectures import MLPArchitecture, GRUArchitecture, CNNArchitecture
        arch_map = {
            "mlp": MLPArchitecture,
            "gru": GRUArchitecture,
            "cnn": CNNArchitecture,
        }
        arch_class = arch_map.get(archi_params.get("core_arch", "mlp"), MLPArchitecture)
        architecture = arch_class(layers=archi_params.get("encode_layer", "128-128"))

        # --- Entraînement (Ray démarre et s'arrête dans ce subprocess) ---
        algo = algo_class(architecture=architecture, hyperparams=algo_params)
        algo.train(
            env_name=env_name,
            map_name=map_name,
            architecture=architecture,
            env_kwargs=env_kwargs,
            stop_criteria=stop_criteria,
            GPUs=GPUs,
            Checkpoints_freq=Checkpoints_freq,
        )

        # --- Lecture du score depuis result.json ---
        score = _read_score_from_filesystem(map_name, trial, training_iterations)

        study.tell(trial, score)
        print(f"[MARIO HPO][PID {worker_id}] Trial #{trial.number} terminé → score={score:.4f}")
        return score

    except optuna.exceptions.TrialPruned:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        return 0.0
    except Exception as exc:
        print(f"[MARIO HPO][PID {os.getpid()}] Trial #{trial.number} échoué : {exc}")
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return 0.0


def _read_score_from_filesystem(
    map_name: str,
    trial: optuna.Trial,
    training_iterations: int,
) -> float:
    """
    Extrait le score ``episode_reward_mean`` depuis le ``result.json`` le plus récent.

    Parcourt le dossier ``exp_results/`` à la racine du projet pour trouver
    le fichier de résultat MARLlib correspondant au run le plus récent.
    Rapporte également le score à Optuna pour le pruning intermédiaire.

    Args:
        map_name (str): Nom du scénario, utilisé pour filtrer les dossiers de résultats.
        trial (optuna.Trial): Trial Optuna courant (pour ``trial.report()``
            et ``trial.should_prune()``).
        training_iterations (int): Nombre d'itérations, utilisé comme ``step``
            dans ``trial.report()``.

    Returns:
        float: Valeur de ``episode_reward_mean`` extraite de la dernière ligne
            de ``result.json``, ou ``0.0`` si le fichier est absent ou illisible.

    Raises:
        optuna.exceptions.TrialPruned: Si Optuna décide d'élaguer ce trial
            après ``trial.report()``.
    """
    src_dir = Path(__file__).resolve().parent.parent.parent
    pattern = str(
        src_dir / "exp_results"
        / f"mappo_*_{map_name}"
        / "MAPPOTrainer_*"
        / "result.json"
    )
    result_files = glob.glob(pattern)

    if not result_files:
        print(f"[MARIO HPO] Aucun result.json trouvé pour map={map_name}, score=0.0")
        return 0.0

    latest = max(result_files, key=os.path.getctime)
    try:
        with open(latest, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
            last = json.loads(lines[-1])
            score = last.get("episode_reward_mean", 0.0)

        trial.report(score, step=training_iterations)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return score

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"[MARIO HPO] Erreur lecture score : {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class HPOptimizer:
    """
    Optimiseur HPO parallèle multi-process pour MARIO + MARLlib.

    Lance ``n_workers`` trials Optuna simultanément via ``joblib``, chacun dans
    un subprocess isolé disposant de son propre contexte Ray. La synchronisation
    est assurée par un storage SQLite partagé.

    Compatible avec tout algorithme héritant de ``Algo`` et tout espace de
    recherche héritant de ``AlgoHyperparametersResearchSpace`` /
    ``ArchiHyperparametersResearchSpace``.

    Attributes:
        algo_class (Type[Algo]): Classe de l'algorithme à optimiser.
        algo_space (AlgoHyperparametersResearchSpace): Espace de recherche algo.
        archi_space (ArchiHyperparametersResearchSpace): Espace de recherche archi.
        env_name (str): Identifiant de l'environnement MARLlib.
        map_name (str): Nom du scénario.
        env_kwargs (dict): Paramètres dynamiques de l'environnement.
        n_trials (int): Nombre total de trials à effectuer.
        n_workers (Optional[int]): Nombre de trials exécutés en parallèle.
        training_iterations (int): Itérations d'entraînement par trial.
        direction (str): ``"maximize"`` ou ``"minimize"``.
        storage_url (str): URL SQLite utilisée par le storage partagé.
        study_name (str): Nom de l'étude Optuna.
        best_policy (JointPolicy | None): Meilleure politique observée.
        study (optuna.Study | None): Objet étude après optimisation.
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
        n_workers: Optional[int] = None,
        training_iterations: int = 5,
        direction: str = "maximize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        stop_criteria: dict = None,
        GPUs: int = 0,
        Checkpoints_freq: int = 1,
    ):
        """
        Configure l'optimiseur HPO parallèle.

        Args:
            algo_class (Type[Algo]): Classe de l'algorithme à instancier pour chaque trial.
            algo_space (AlgoHyperparametersResearchSpace): Espace de recherche des
                hyperparamètres de l'algorithme (ex: ``PPOAlgoSpace()``).
            archi_space (ArchiHyperparametersResearchSpace): Espace de recherche de
                l'architecture réseau (ex: ``MLPArchiSpace()``).
            env_name (str): Identifiant de l'environnement MARLlib (ex: ``"mpe"``).
            map_name (str): Nom du scénario (ex: ``"simple_adversary_v3"``).
            env_kwargs (dict, optional): Paramètres dynamiques de l'environnement.
            n_trials (int): Nombre total de trials à effectuer. Défaut : ``10``.
            n_workers (Optional[int]): Nombre de trials exécutés simultanément.
                ``None`` (défaut) = autant de workers que de trials (tous en parallèle).
                ``1`` = séquentiel. Toute valeur intermédiaire limite le parallélisme.
            training_iterations (int): Nombre d'itérations d'entraînement par trial.
                Défaut : ``5``.
            direction (str): Sens d'optimisation — ``"maximize"`` (récompense)
                ou ``"minimize"`` (perte). Défaut : ``"maximize"``.
            pruner (optuna.pruners.BasePruner, optional): Stratégie d'élagage des
                trials peu prometteurs. Défaut : ``MedianPruner(n_startup_trials=3)``.
            sampler (optuna.samplers.BaseSampler, optional): Stratégie d'échantillonnage.
                Défaut : ``TPESampler()``.
            study_name (str, optional): Nom de l'étude Optuna. Généré automatiquement
                si non fourni.
            storage_path (str, optional): Chemin du fichier SQLite de stockage partagé.
                Défaut : ``<racine_projet>/exp_results/optuna_storage.db``.
            stop_criteria (dict, optional): Conditions d'arrêt transmises à
                ``algo.train()``. Défaut : ``{"training_iteration": training_iterations}``.
            GPUs (int): Nombre de GPUs alloués par trial. Défaut : ``0``.
            Checkpoints_freq (int): Fréquence de sauvegarde des checkpoints. Défaut : ``1``.
        """
        self.algo_class = algo_class
        self.algo_space = algo_space
        self.archi_space = archi_space
        self.env_name = env_name
        self.map_name = map_name
        self.env_kwargs = env_kwargs or {}
        self.n_trials = n_trials
        self.n_workers = n_workers if n_workers is not None else n_trials
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

        # Storage SQLite partagé entre tous les workers
        if storage_path is None:
            src_dir = Path(__file__).resolve().parent.parent.parent
            db_path = src_dir / "exp_results" / "optuna_storage.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path = str(db_path)
        self.storage_url = f"sqlite:///{storage_path}"

    # ------------------------------------------------------------------
    # Méthode publique principale
    # ------------------------------------------------------------------

    def optimize(self) -> tuple:
        """
        Lance la recherche HPO parallèle et retourne la meilleure politique.

        Crée (ou charge) l'étude Optuna dans le storage SQLite, distribue
        ``n_trials`` trials en batches de ``n_workers`` via ``joblib``, puis
        reconstruit la meilleure ``MARLlibPolicy`` depuis le filesystem.

        Returns:
            tuple: ``(best_policy, study)`` où ``best_policy`` est une instance
                de ``MARLlibPolicy`` (ou ``None`` si aucun trial n'a abouti) et
                ``study`` est l'objet ``optuna.Study`` complet.

        Raises:
            RuntimeError: Si ``joblib`` n'est pas installé.
        """
        # Création ou rechargement de l'étude dans le storage partagé
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True,   # permet la reprise après interruption
        )

        effective_workers = min(self.n_workers, self.n_trials)

        print(f"\n[MARIO HPO] Démarrage : {self.study_name}")
        print(f"  Essais totaux : {self.n_trials}")
        print(f"  Workers       : {effective_workers} (parallèle)" if effective_workers > 1
              else f"  Workers       : 1 (séquentiel)")
        print(f"  Direction     : {self.direction}")
        print(f"  Itérations    : {self.training_iterations} par essai")
        print(f"  Storage       : {self.storage_url}\n")

        # Paramètres communs sérialisés pour chaque subprocess
        worker_kwargs = dict(
            study_name=self.study_name,
            storage_url=self.storage_url,
            algo_class_name=self.algo_class.__name__,
            algo_module=self.algo_class.__module__,
            algo_space_class_name=self.algo_space.__class__.__name__,
            algo_space_module=self.algo_space.__class__.__module__,
            archi_space_class_name=self.archi_space.__class__.__name__,
            archi_space_module=self.archi_space.__class__.__module__,
            env_name=self.env_name,
            map_name=self.map_name,
            env_kwargs=self.env_kwargs,
            stop_criteria=self.stop_criteria,
            training_iterations=self.training_iterations,
            GPUs=self.GPUs,
            Checkpoints_freq=self.Checkpoints_freq,
            direction=self.direction,
        )

        # Distribution des trials en batches de n_workers
        # Chaque appel à _worker_trial demande lui-même son trial au storage
        joblib.Parallel(n_jobs=effective_workers, backend="loky")(
            joblib.delayed(_worker_trial)(**worker_kwargs)
            for _ in range(self.n_trials)
        )

        # Rechargement de l'étude depuis le storage pour avoir les résultats à jour
        self.study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage_url,
        )

        return self._finalize()

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _finalize(self) -> tuple:
        """
        Analyse les résultats de l'étude et reconstruit la meilleure politique.

        Affiche le résumé de l'optimisation, puis reconstruit une instance de
        ``MARLlibPolicy`` en localisant le checkpoint correspondant au meilleur
        trial dans le filesystem (dossier ``exp_results/``).

        Returns:
            tuple: ``(best_policy, study)``.
        """
        print("\n[MARIO HPO] Optimisation terminée !")

        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed:
            print("[MARIO HPO] Aucun essai complété — vérifier les erreurs ci-dessus.")
            return None, self.study

        best = self.study.best_trial
        print(f"  Trials complétés : {len(completed)} / {self.n_trials}")
        print(f"  Meilleur trial   : #{best.number}")
        print(f"  Meilleur score   : {self.study.best_value:.4f}")
        print(f"  Meilleurs params : {self.study.best_params}")

        best_policy = self._rebuild_best_policy(best)
        return best_policy, self.study

    def _rebuild_best_policy(self, best_trial: optuna.Trial) -> Optional[JointPolicy]:
        """
        Reconstruit une ``MARLlibPolicy`` depuis le run correspondant au meilleur trial.

        Localise le dossier ``exp_results/`` à la racine du projet, sélectionne
        le run MARLlib le plus récent correspondant au ``map_name``, et instancie
        une ``MARLlibPolicy`` avec le modèle et l'algorithme reconstruits depuis
        les paramètres du meilleur trial.

        Args:
            best_trial (optuna.Trial): Meilleur trial Optuna, utilisé pour
                reconstruire l'architecture et les hyperparamètres.

        Returns:
            MARLlibPolicy | None: Instance de la meilleure politique, ou ``None``
                si aucun run MARLlib n'est trouvé sur le filesystem.
        """
        from mario.algos.marllibpolicy import MARLlibPolicy
        from mario.algos.architectures import MLPArchitecture, GRUArchitecture, CNNArchitecture

        src_dir = Path(__file__).resolve().parent.parent.parent
        pattern = str(
            src_dir / "exp_results"
            / f"mappo_*_{self.map_name}"
            / "MAPPOTrainer_*"
        )
        run_dirs = glob.glob(pattern)

        if not run_dirs:
            print("[MARIO HPO] Aucun run trouvé sur le filesystem pour reconstruire la policy.")
            return None

        best_run_dir = max(run_dirs, key=os.path.getctime)
        print(f"[MARIO HPO] Run sélectionné pour la policy : {best_run_dir}")

        # Reconstruction de l'architecture depuis les params du meilleur trial
        params = best_trial.params
        arch_map = {"mlp": MLPArchitecture, "gru": GRUArchitecture, "cnn": CNNArchitecture}
        arch_type = params.get("core_arch", "mlp")
        arch_class = arch_map.get(arch_type, MLPArchitecture)

        width = params.get("layer_width", 128)
        depth = params.get("num_layers", 2)
        layers = "-".join([str(width)] * depth)
        architecture = arch_class(layers=layers)

        algo_params = {
            k: v for k, v in params.items()
            if k not in ("core_arch", "layer_width", "num_layers")
        }
        algo_instance = self.algo_class(architecture=architecture, hyperparams=algo_params)

        arch_label = architecture.type.lower()
        exp_pattern = f"mappo_{arch_label}_{self.map_name}/MAPPOTrainer_*"

        from marllib import marl
        env_kwargs_copy = dict(self.env_kwargs)
        env = marl.make_env(
            environment_name=self.env_name,
            map_name=self.map_name,
            **env_kwargs_copy,
        )

        # Récupération de l'objet algo MARLlib interne
        marllib_algo = marl.algos.mappo(hyperparam_source=self.env_name)
        arch_config = architecture.to_marllib_config()
        model = marl.build_model(env, marllib_algo, arch_config)

        return MARLlibPolicy(model, marllib_algo, env, exp_pattern)