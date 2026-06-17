"""
# `marllibpolicy` — Politique MARLlib pour MARIO

Ce module fournit la classe `MARLlibPolicy`, point d'articulation entre
les modèles entraînés par **MARLlib** et l'interface de décision abstraite
définie par le moteur **MARIO**.

## Dépendances système

Les versions suivantes sont requises pour le rendu visuel avec PettingZoo MPE :

```bash
pip install "pyglet==1.5.27" "Pillow==9.5.0"
```

## Exemple d'utilisation

```python
from mario.algos.ppo import PPOAlgo
from mario.algos.architectures import MLPArchitecture

algo = PPOAlgo(architecture=MLPArchitecture(layers="128-128"))
policy = algo.train(env_name="mpe", map_name="simple_world_comm")

# Rendu visuel en fenêtre
policy.render(save_mode="human")

# Sauvegarde en vidéo MP4
policy.render(save_mode="video")
```
"""
import os
import csv
import glob
import json
from marllib import marl
from pathlib import Path
from mario.algos.policies import JointPolicy


def patch_marllib():
    """
    Applique un monkey-patch sur MARLlib pour autoriser l'injection de paramètres d'environnement dynamiques.

    Par défaut, MARLlib utilise une fonction interne `dict_update` dotée d'une vérification stricte.
    Celle-ci rejette systématiquement tout argument (ex: `num_good`, `num_adversaries`) qui n'est pas
    explicitement déclaré dans ses propres fichiers de configuration `.yaml`.

    Ce patch remplace cette fonction à l'exécution par une version tolérante, permettant à
    l'utilisateur de modifier la configuration de l'environnement à la volée (via `**kwargs`)
    sans lever de `ValueError`.

    **Mécanisme d'action :**
    Pour contourner les problèmes de liaison d'import (Import Binding), la fonction écrase
    la référence stricte de `dict_update` dans tous les espaces de noms (namespaces) critiques :

    1. Le module source d'origine (`marllib.marl.common`).
    2. L'espace de noms d'initialisation (`marllib.marl`), qui est la référence copiée
       et utilisée par `make_env()` de `__init__.py`.
    3. Le sous-module `marllib.marl.marl` par mesure de sécurité.

    Notes:
        Cette fonction doit impérativement être appelée **avant** toute instanciation
        d'environnement (notamment juste avant `marl.make_env()`) pour s'assurer que le
        patch écrase bien les références en mémoire au bon moment.

    Raises:
        ImportError: L'exception est interceptée en interne de manière silencieuse si
            le module `marllib` n'est pas installé ou introuvable. Un avertissement
            est simplement affiché dans la console et l'exécution se poursuit.
    """
    def tolerant_dict_update(d, u, *args, **kwargs):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = tolerant_dict_update(d[k], v)
            else:
                d[k] = v
        return d

    try:
        import marllib
        import marllib.marl.common

        marllib.marl.common.dict_update = tolerant_dict_update
        marllib.marl.dict_update = tolerant_dict_update

        try:
            import marllib.marl.marl
            marllib.marl.marl.dict_update = tolerant_dict_update
        except ImportError:
            pass

        print("[MARIO] Monkey-patch MARLlib appliqué (Namespaces écrasés avec succès).")
    except ImportError as e:
        print(f"[MARIO] [Attention] Impossible d'appliquer le patch MARLlib : {e}")


class MARLlibPolicy(JointPolicy):
    """
    Politique entraînée par MARLlib, compatible avec l'interface MARIO.

    Cette classe encapsule le modèle, l'algorithme et l'environnement produits
    lors d'un entraînement MARLlib, et expose les méthodes `predict()` et
    `render()` attendues par le moteur MARIO.

    Attributes:
        model: Objet modèle (réseau de neurones) généré par `marl.build_model()`.
        algo: Instance de l'algorithme MARLlib (ex : objet `mappo` retourné
            par `marl.algos.mappo()`).
        env: Environnement MARLlib retourné par `marl.make_env()` lors de
            l'entraînement.
        exp_pattern (str): Patron glob relatif au dossier `exp_results/` pour
            localiser les résultats du run, par exemple
            `"mappo_mlp_simple_world_comm/MAPPOTrainer_*"`.

    Note:
        `pyglet==1.5.27` et `Pillow==9.5.0` sont requis pour le rendu visuel.
    """

    def __init__(self, model, algo_instance, env_instance, exp_pattern):
        """Initialise le conteneur de la politique entraînée.

        Args:
            model: Objet réseau de neurones généré par MARLlib.
            algo_instance: Instance de l'algorithme ayant servi à l'apprentissage
                (objet MARLlib, *pas* une instance de `PPOAlgo`).
            env_instance: Environnement retourné par `marl.make_env()`.
            exp_pattern (str): Patron glob relatif pour retrouver les résultats,
                ex. `"mappo_mlp_simple_world_comm/MAPPOTrainer_*"`.
        """
        super().__init__(policy_type="MARLlib_Policy")
        self.model = model
        self.algo = algo_instance
        self.env = env_instance
        self.exp_pattern = exp_pattern

    def predict(self, observations):
        """Placeholder — à implémenter."""
        return {}

    # ------------------------------------------------------------------
    # RENDER — point d'entrée principal
    # ------------------------------------------------------------------
    def render(
        self,
        env=None,
        model=None,
        save_mode: str = "human",
        run_dir: str = None,
        checkpoint: "int | str | None" = None,
    ):
        """Rejoue la politique entraînée à partir du checkpoint le plus récent.

        Localise automatiquement le dossier `exp_results/` en remontant
        l'arborescence depuis ce fichier, sélectionne le run le plus récent
        correspondant à `exp_pattern`, puis charge le checkpoint le plus avancé.

        L'environnement est recréé depuis `params.json` pour garantir la
        cohérence avec la configuration d'entraînement.

        Args:
            env: Ignoré (recréé automatiquement depuis `params.json`).
                Conservé pour compatibilité avec l'interface MARIO.
            model: Modèle à utiliser. Si `None`, utilise `self.model`.
            save_mode (str): Mode de sortie du rendu. Valeurs acceptées :

                - `"human"` *(défaut)* — affichage dans une fenêtre graphique.
                - `"video"` — sauvegarde en fichier `.mp4`
                  (nécessite `pip install imageio imageio-ffmpeg`).
                - `"gif"` — sauvegarde en fichier `.gif`
                  (nécessite `pip install imageio`).

        Raises:
            FileNotFoundError: Si aucun dossier `exp_results/` n'est trouvé,
                si aucun run ne correspond au `exp_pattern`, ou si aucun
                checkpoint n'est présent dans le run sélectionné.
            ValueError: Si `save_mode` n'est pas `"human"`, `"video"` ou `"gif"`.
            ImportError: Si `imageio` (et `imageio-ffmpeg` pour MP4) n'est pas
                installé lors d'un rendu `"video"` ou `"gif"`.

        Note:
            Les fichiers vidéo/GIF sont sauvegardés dans un sous-dossier
            `renders/` créé automatiquement à l'intérieur du dossier du run.
            Pour le moment l'enregistrment est impossible du à des conflit avec MARLlib et RLlib.

        Example:
            ```python
            policy.render()                      # fenêtre temps réel
            policy.render(save_mode="video")     # → renders/render_checkpoint_10.mp4
            policy.render(save_mode="gif")       # → renders/render_checkpoint_10.gif
            ```
        """
        patch_marllib()

        base_path = run_dir or self._choose_run_dir()
        print(f"[MARIO] Run sélectionné : {base_path}")

        # Reconstruit ENV + MODEL depuis les params du run sélectionné,
        # pour garantir la cohérence avec le checkpoint choisi.
        env, model = self._recreate_env_and_model(base_path)

        checkpoint_dir, checkpoint_num = self._select_checkpoint(base_path, checkpoint)
        checkpoint_file = str(Path(checkpoint_dir) / f"checkpoint-{checkpoint_num}")
        print(f"[MARIO] Checkpoint utilisé : checkpoint-{checkpoint_num}")

        output_path = self._build_output_path(base_path, checkpoint_num, save_mode)
        render_config = self._build_render_config(save_mode, output_path)
        print(f"[MARIO] Mode rendu : {save_mode} | Sortie : {output_path}")

        self.algo.render(
            env, model,
            restore_path={
                'params_path': str(Path(base_path) / "params.json"),
                'model_path': checkpoint_file,
                **render_config,
            },
            local_mode=True,
            num_gpus=0,
            share_policy="group",
            checkpoint_end=True,
            evaluation_interval=None,
            stop_timesteps=1,
            timesteps_total=1,
            stop_iters=1,
        )

        if save_mode == "gif":
            record_dir = render_config["record_env"]
            self._convert_last_mp4_to_gif(record_dir, output_path)

    # ------------------------------------------------------------------
    # Méthodes internes — chargement / chemins
    # ------------------------------------------------------------------
    def _load_params(self, base_path: Path) -> dict:
        """Charge et retourne le contenu de `params.json` d'un run.

        Désactive également `evaluation_interval` dans le dict retourné afin
        d'éviter un crash du worker Ray lors du rendu (l'évaluation nécessite
        un environnement supplémentaire incompatible avec le mode rendu).

        Args:
            base_path (Path): Chemin vers le dossier racine du run MARLlib
                (dossier contenant `params.json`).

        Returns:
            dict: Contenu de `params.json` avec `evaluation_interval` mis à
                `None` aux niveaux racine et `custom_model_config`.

        Raises:
            FileNotFoundError: Si `params.json` est absent de `base_path`.
        """
        params_path = Path(base_path) / "params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"params.json introuvable : {params_path}")
        with open(params_path, "r") as f:
            params = json.load(f)
        params["evaluation_interval"] = None
        params["model"]["custom_model_config"]["evaluation_interval"] = None
        return params

    def _find_exp_results_dir(self) -> Path:
        """Localise le dossier `exp_results/` en remontant l'arborescence.

        Remonte récursivement depuis le répertoire de ce fichier jusqu'à la
        racine du système de fichiers (`/`), et retourne le premier dossier
        `exp_results/` trouvé.

        Returns:
            Path: Chemin absolu vers le dossier `exp_results/`.

        Raises:
            FileNotFoundError: Si aucun dossier `exp_results/` n'est trouvé
                avant d'atteindre la racine du système de fichiers.
        """
        src_dir = Path(__file__).resolve().parent.parent.parent
        exp_results = src_dir / "exp_results"
        exp_results.mkdir(exist_ok=True)
        return exp_results

    def _recreate_env_and_model(self, base_path: Path):
        """Recrée l'environnement ET le modèle depuis params.json du run choisi.

        Args:
            base_path (Path): Chemin vers le dossier racine du run contenant
                `params.json`.

        Returns:
            tuple: `(env_output, model)` où `env_output` est le tuple brut
                `(env_instance, info)` retourné par `marl.make_env()`. Ce
                tuple ne doit PAS être dégroupé : `marl.build_model()` accède
                lui-même à `environment[0]` en interne, et `algo.render()` /
                `algo.fit()` font `env_instance, info = env`. Le tuple complet
                doit donc être transmis tel quel aux deux appels.
        """
        params = self._load_params(base_path)
        custom_cfg = params["model"]["custom_model_config"]

        env_args = custom_cfg["env_args"].copy()
        env_name = custom_cfg["env"]
        map_name = env_args.pop("map_name")

        print(f"[MARIO] Env recréé depuis params.json : {env_name}:{map_name}")
        env_output = marl.make_env(environment_name=env_name, map_name=map_name, **env_args)

        model_arch_args = custom_cfg.get("model_arch_args", {})
        arch_config = model_arch_args if model_arch_args else {"core_arch": "mlp", "encode_layer": "128-128"}

        print(f"[MARIO] Architecture reconstruite depuis params.json : {arch_config}")

        # build_model attend le tuple complet (env_instance, info), pas env_instance seul
        model = marl.build_model(env_output, self.algo, arch_config)

        # render()/fit() attendent eux aussi le tuple complet
        return env_output, model

    # ------------------------------------------------------------------
    # Méthodes internes — sélection du run
    # ------------------------------------------------------------------
    def _list_run_dirs(self) -> list:
        """Liste tous les dossiers de run correspondant à `self.exp_pattern`.

        Recherche d'abord dans le dossier `exp_results/` localisé via
        `_find_exp_results_dir()`, en utilisant `self.exp_pattern` comme
        motif glob (ex. `"mappo_mlp_simple_world_comm/MAPPOTrainer_*"`).
        Si aucun résultat n'est trouvé à cet emplacement, effectue une
        recherche récursive (`rglob`) en remontant depuis le répertoire de
        ce fichier jusqu'à la racine du système de fichiers, afin de
        retrouver des runs potentiellement situés ailleurs (utile par
        exemple pour des trials issus d'une session HPO).

        Returns:
            list[str]: Liste des chemins de run trouvés, triés du plus
                récent au plus ancien (par date de création). Liste vide
                si aucun run n'est trouvé.
        """
        exp_results_dir = self._find_exp_results_dir()
        pattern = str(exp_results_dir / self.exp_pattern)
        run_dirs = glob.glob(pattern)

        if not run_dirs:
            project_root = Path(__file__).resolve().parent
            while project_root != project_root.parent:
                project_root = project_root.parent
                all_results = list(project_root.rglob(self.exp_pattern))
                if all_results:
                    run_dirs = [str(p) for p in all_results]
                    break

        return sorted(run_dirs, key=os.path.getctime, reverse=True)

    def _choose_run_dir(self) -> str:
        """Sélectionne un run parmi ceux disponibles, de manière interactive si nécessaire.

        Appelle `_list_run_dirs()` pour obtenir la liste des runs
        correspondant à `self.exp_pattern`. Si un seul run est trouvé, il
        est retourné directement sans interaction. Si plusieurs runs sont
        disponibles (par exemple plusieurs trials d'une session HPO),
        affiche la liste dans la console et invite l'utilisateur à choisir
        un index ; une entrée vide sélectionne le run le plus récent
        (index `0`) par défaut.

        Returns:
            str: Chemin du run sélectionné.

        Raises:
            FileNotFoundError: Si aucun run ne correspond à `self.exp_pattern`.
        """
        run_dirs = self._list_run_dirs()

        if not run_dirs:
            raise FileNotFoundError(
                f"Aucun résultat trouvé.\nPattern : {self.exp_pattern}"
            )

        if len(run_dirs) == 1:
            return run_dirs[0]

        print("\n[MARIO] Plusieurs runs disponibles :")
        for i, d in enumerate(run_dirs):
            print(f"  [{i}] {d}")

        while True:
            choice = input(f"Choisir un run [0-{len(run_dirs) - 1}] (défaut: 0) : ").strip()
            if choice == "":
                return run_dirs[0]
            if choice.isdigit() and 0 <= int(choice) < len(run_dirs):
                return run_dirs[int(choice)]
            print("Choix invalide, réessayez.")

    # ------------------------------------------------------------------
    # Méthodes internes — sélection du checkpoint
    # ------------------------------------------------------------------
    def _select_checkpoint(self, base_path: str, checkpoint: "int | str | None"):
        """Résout le critère de sélection en un checkpoint concret.

        Centralise la logique de résolution du paramètre `checkpoint` de
        `render()` en un dossier de checkpoint précis. Le comportement
        dépend du type/valeur de `checkpoint` :

        - `None` : sélection interactive parmi tous les checkpoints du run
          (voir `_choose_checkpoint_dir`).
        - `"last"` : le checkpoint le plus avancé (numéro le plus élevé).
        - `"best"` : le checkpoint correspondant à la meilleure
          `episode_reward_mean` trouvée dans `progress.csv` (voir
          `_best_checkpoint_dir`).
        - `int` : un numéro de checkpoint précis (ex. `20` pour
          `checkpoint_20`).

        Args:
            base_path (str): Chemin du dossier racine du run, contenant
                les sous-dossiers `checkpoint_*`.
            checkpoint (int | str | None): Critère de sélection du
                checkpoint, voir description ci-dessus.

        Returns:
            tuple[str, int]: Un couple `(checkpoint_dir, checkpoint_num)`
                où `checkpoint_dir` est le chemin du dossier checkpoint
                sélectionné et `checkpoint_num` son numéro entier.

        Raises:
            FileNotFoundError: Si `base_path` ne contient aucun dossier
                `checkpoint_*`, ou si le numéro de checkpoint demandé
                explicitement n'existe pas parmi les checkpoints
                disponibles.
        """
        checkpoint_dirs = sorted(
            glob.glob(str(Path(base_path) / "checkpoint_*")),
            key=lambda p: int(Path(p).name.split("_")[-1])
        )
        if not checkpoint_dirs:
            raise FileNotFoundError(f"Aucun checkpoint dans : {base_path}")

        if checkpoint is None:
            chosen = self._choose_checkpoint_dir(checkpoint_dirs)
        elif checkpoint == "last":
            chosen = checkpoint_dirs[-1]
        elif checkpoint == "best":
            chosen = self._best_checkpoint_dir(base_path, checkpoint_dirs)
        else:
            target = f"checkpoint_{int(checkpoint)}"
            matches = [d for d in checkpoint_dirs if Path(d).name == target]
            if not matches:
                raise FileNotFoundError(
                    f"Checkpoint {checkpoint} introuvable dans {base_path}. "
                    f"Disponibles : {[Path(d).name for d in checkpoint_dirs]}"
                )
            chosen = matches[0]

        num = int(Path(chosen).name.split("_")[-1])
        return chosen, num

    def _choose_checkpoint_dir(self, checkpoint_dirs: list) -> str:
        """Sélection interactive d'un checkpoint parmi ceux disponibles.

        Affiche dans la console la liste des checkpoints disponibles
        (triés par numéro croissant), en signalant le dernier comme valeur
        par défaut, puis invite l'utilisateur à choisir un index. Une
        entrée vide sélectionne automatiquement le checkpoint le plus
        avancé.

        Args:
            checkpoint_dirs (list[str]): Liste des chemins de dossiers
                `checkpoint_*`, triés par numéro de checkpoint croissant.

        Returns:
            str: Chemin du dossier checkpoint sélectionné.
        """
        print("\n[MARIO] Checkpoints disponibles :")
        for i, d in enumerate(checkpoint_dirs):
            num = Path(d).name.split("_")[-1]
            marker = " (dernier)" if i == len(checkpoint_dirs) - 1 else ""
            print(f"  [{i}] checkpoint_{num}{marker}")

        default_idx = len(checkpoint_dirs) - 1
        while True:
            choice = input(
                f"Choisir un checkpoint [0-{default_idx}] (défaut: {default_idx}) : "
            ).strip()
            if choice == "":
                return checkpoint_dirs[default_idx]
            if choice.isdigit() and 0 <= int(choice) < len(checkpoint_dirs):
                return checkpoint_dirs[int(choice)]
            print("Choix invalide, réessayez.")

    def _best_checkpoint_dir(self, base_path: str, checkpoint_dirs: list) -> str:
        """Détermine le checkpoint correspondant à la meilleure performance.

        Lit le fichier `progress.csv` généré par Ray Tune dans `base_path`
        et recherche, parmi les itérations enregistrées, celle présentant
        la meilleure valeur de `episode_reward_mean`. Le checkpoint
        sélectionné est le plus avancé parmi ceux dont le numéro est
        inférieur ou égal à cette itération.

        Si `progress.csv` est absent, vide ou ne contient pas de colonnes
        exploitables (`training_iteration`, `episode_reward_mean`),
        retombe silencieusement sur le comportement `"last"`.

        Args:
            base_path (str): Chemin du dossier racine du run, contenant
                potentiellement `progress.csv`.
            checkpoint_dirs (list[str]): Liste des chemins de dossiers
                `checkpoint_*` disponibles pour ce run, triés par numéro
                croissant.

        Returns:
            str: Chemin du dossier checkpoint jugé "meilleur", ou le
                dernier checkpoint disponible en cas de fallback.
        """
        progress_path = Path(base_path) / "progress.csv"
        if not progress_path.exists():
            print("[MARIO] progress.csv introuvable, fallback sur 'last'.")
            return checkpoint_dirs[-1]

        rows = []
        with open(progress_path, "r") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append((
                        int(row["training_iteration"]),
                        float(row.get("episode_reward_mean", "nan"))
                    ))
                except (KeyError, ValueError):
                    continue

        if not rows:
            print("[MARIO] progress.csv vide/illisible, fallback sur 'last'.")
            return checkpoint_dirs[-1]

        best_iter, _ = max(rows, key=lambda r: r[1])
        candidates = [
            d for d in checkpoint_dirs
            if int(Path(d).name.split("_")[-1]) <= best_iter
        ]
        return candidates[-1] if candidates else checkpoint_dirs[-1]

    # ------------------------------------------------------------------
    # Méthodes internes — sortie gif/vidéo
    # ------------------------------------------------------------------
    def _build_output_path(self, base_path: str, checkpoint_num: int, save_mode: str) -> str:
        """Construit le chemin de sortie centralisé pour les exports gif/vidéo.

        Place les fichiers générés dans un emplacement centralisé sous
        `exp_results/rendered/gif/` (pour `save_mode="gif"`) ou
        `exp_results/rendered/video/` (pour `save_mode="video"`), plutôt
        que dans le dossier du run lui-même. Le nom de fichier inclut le
        nom du run et le numéro de checkpoint afin d'éviter toute
        collision entre plusieurs runs ou checkpoints rendus.

        Args:
            base_path (str): Chemin du dossier racine du run, utilisé
                uniquement pour en extraire le nom (`Path(base_path).name`)
                et l'inclure dans le nom de fichier de sortie.
            checkpoint_num (int): Numéro du checkpoint rendu, inclus dans
                le nom de fichier.
            save_mode (str): Mode de rendu — `"human"`, `"video"` ou
                `"gif"`. Pour `"human"`, aucun fichier n'étant généré, une
                chaîne vide est retournée.

        Returns:
            str: Chemin de sortie **sans extension** (l'extension `.mp4`
                ou `.gif` est ajoutée ultérieurement par
                `_build_render_config`/`_convert_last_mp4_to_gif`). Chaîne
                vide si `save_mode == "human"`.
        """
        if save_mode == "human":
            return ""

        exp_results_dir = self._find_exp_results_dir()
        subfolder = "gif" if save_mode == "gif" else "video"
        output_dir = exp_results_dir / "rendered" / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        run_name = Path(base_path).name
        filename = f"{run_name}_checkpoint_{checkpoint_num}"
        return str(output_dir / filename)

    def _build_render_config(self, save_mode: str, output_path: str) -> dict:
        """Construit `render_env`/`record_env`, les clés réellement
        comprises par MARLlib/RLlib (et non `video_dir`/`gif_dir`,
        qui n'existent pas côté RLlib).
        """
        if save_mode == "human":
            return {
                "render_env": True,
                "record_env": False,
            }

        elif save_mode == "video":
            record_dir = str(Path(output_path).parent)
            return {
                "render_env": False,
                "record_env": record_dir,
            }

        elif save_mode == "gif":
            try:
                import imageio  # noqa: F401
            except ImportError:
                raise ImportError("pip install imageio imageio-ffmpeg")
            record_dir = str(Path(output_path).parent)
            return {
                "render_env": False,
                "record_env": record_dir,
            }

        else:
            raise ValueError(f"save_mode inconnu : '{save_mode}'. Choix : 'human', 'video', 'gif'")

    def _convert_last_mp4_to_gif(self, record_dir: str, output_path: str):
        """Convertit le fichier `.mp4` le plus récent en `.gif`.

        MARLlib/RLlib ne savent générer que des vidéos `.mp4` via le
        paramètre `record_env` (le wrapper Gym `RecordVideo` sous-jacent ne
        supporte pas l'export direct en GIF). Cette méthode comble cet écart
        en post-traitement : elle recherche dans `record_dir` le fichier
        `.mp4` le plus récemment créé, puis le convertit en `.gif` à l'aide
        d'`imageio`, en conservant le FPS d'origine de la vidéo lorsque celui-
        ci est disponible dans ses métadonnées.

        Args:
            record_dir (str): Dossier dans lequel RLlib a enregistré le(s)
                fichier(s) `.mp4` lors du rendu (correspond à la valeur de
                `record_env` passée à `self.algo.render()`).
            output_path (str): Chemin de sortie **sans extension** pour le
                GIF final. L'extension `.gif` est ajoutée automatiquement.

        Returns:
            None: Le GIF est écrit sur disque à `output_path + ".gif"`; rien
                n'est retourné.

        Raises:
            FileNotFoundError: Si aucun fichier `.mp4` n'est trouvé dans
                `record_dir`.
            ImportError: Si `imageio` n'est pas installé dans l'environnement
                (nécessaire à la lecture du `.mp4` et à l'écriture du `.gif`).

        Note:
            Toutes les frames de la vidéo sont chargées en mémoire avant
            l'écriture du GIF (`[frame for frame in reader]`); pour des
            rendus très longs, cela peut représenter une consommation
            mémoire significative.
        """
        import imageio

        mp4_files = sorted(
            glob.glob(str(Path(record_dir) / "*.mp4")),
            key=os.path.getctime
        )
        if not mp4_files:
            raise FileNotFoundError(f"Aucun .mp4 trouvé dans {record_dir} pour conversion en gif.")

        latest_mp4 = mp4_files[-1]
        gif_path = output_path + ".gif"

        reader = imageio.get_reader(latest_mp4)
        fps = reader.get_meta_data().get("fps", 20)
        frames = [frame for frame in reader]
        imageio.mimsave(gif_path, frames, fps=fps)

        print(f"[MARIO] GIF généré : {gif_path}")