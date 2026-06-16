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
import glob
import json
from marllib import marl
from pathlib import Path
from mario.algos.policies import JointPolicy
# note pour plus tard  "pyglet==1.5.27" à ajouter à la doc
# meme chose pour "Pillow==9.5.0"

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
       et utilisée par `make_env()` à la ligne 102 de `__init__.py`.
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
    def __init__(self, model, algo_instance, env_instance,exp_pattern):
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
        """
        Calcule les actions des agents à partir d'un état d'observation.

        Note: L'implémentation actuelle est un substitut (placeholder).
        À terme, cette méthode exploitera les capacités d'inférence de self.model
        ou les routines de rendu de self.algo.

        Returns:
            dict: Dictionnaire vide dans l'attente de l'implémentation du moteur d'inférence.
        """
        return {}

    def render(self,env=None,model=None,save_mode: str = "human"):
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

        Example:
            ```python
            policy.render()                      # fenêtre temps réel
            policy.render(save_mode="video")     # → renders/render_checkpoint_10.mp4
            policy.render(save_mode="gif")       # → renders/render_checkpoint_10.gif
            ```
        """
        patch_marllib()
        model = model or self.model

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

        if not run_dirs:
            raise FileNotFoundError(
                f"Aucun résultat trouvé.\nPattern : {self.exp_pattern}"
            )

        base_path = max(run_dirs, key=os.path.getctime)
        print(f"[MARIO] Run sélectionné  : {base_path}")

        env = self._recreate_env(base_path)

        checkpoint_dirs = sorted(
            glob.glob(str(Path(base_path) / "checkpoint_*")),
            key=lambda p: int(Path(p).name.split("_")[-1])
        )

        if not checkpoint_dirs:
            raise FileNotFoundError(f"Aucun checkpoint dans : {base_path}")

        latest_checkpoint_dir = checkpoint_dirs[-1]
        checkpoint_num = int(Path(latest_checkpoint_dir).name.split("_")[-1])
        checkpoint_file = str(Path(latest_checkpoint_dir) / f"checkpoint-{checkpoint_num}")

        print(f"[MARIO] Checkpoint utilisé : checkpoint-{checkpoint_num}")

        # --- Gestion des modes de sauvegarde ---
        output_dir = Path(base_path) / "renders"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"render_checkpoint_{checkpoint_num}")

        render_config = self._build_render_config(save_mode, output_path)
        print(f"[MARIO] Mode rendu : {save_mode} | Sortie : {output_path}")

        self.algo.render(env, model,
                     restore_path={
                         'params_path': str(Path(base_path) /"params.json"),
                         'model_path': checkpoint_file,
                         #f"{base_path}/checkpoint_000010/checkpoint-10",
                         'render': True
                         },
                         local_mode=True,
                         num_gpus=0,
                         share_policy="group",
                         checkpoint_end=False,
                         evaluation_interval=None,
                         **render_config
                         )

    # ------------------------------------------------------------------
    # Méthodes internes
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
        exp_results.mkdir(exist_ok=True)  # crée le dossier si absent
        return exp_results

    def _recreate_env(self, base_path: Path):
        """Recrée l'environnement MARLlib depuis `params.json`.

        Extrait `env` (ex. `"mpe"`) et `map_name` (ex. `"simple_world_comm"`)
        depuis `params.json`, puis appelle `marl.make_env()`. Cela garantit
        que l'environnement du rendu est identique à celui de l'entraînement,
        même si le processus courant est différent.

        Args:
            base_path (Path): Chemin vers le dossier racine du run contenant
                `params.json`.

        Returns:
            Environnement MARLlib enregistré auprès de Ray Tune.
        """
        params = self._load_params(base_path)
        env_args = params["model"]["custom_model_config"]["env_args"].copy()
        env_name = params["model"]["custom_model_config"]["env"]
        map_name = env_args.pop("map_name")

        print(f"[MARIO] Env recréé depuis params.json : {env_name}:{map_name}")
        return marl.make_env(environment_name=env_name, map_name=map_name, **env_args)

    def _build_render_config(self, save_mode: str, output_path: str) -> dict:
        """Construit le dictionnaire de configuration pour le mode de rendu choisi.

        Args:
            save_mode (str): Mode de rendu — `"human"`, `"video"` ou `"gif"`.
            output_path (str): Chemin de sortie **sans extension** pour les modes
                `"video"` et `"gif"`. L'extension (`.mp4` ou `.gif`) est ajoutée
                automatiquement.

        Returns:
            dict: Dictionnaire de paramètres à passer en `**kwargs` à
                `self.algo.render()`. Vide pour `"human"`.

        Raises:
            ValueError: Si `save_mode` ne vaut ni `"human"`, `"video"`, ni `"gif"`.
            ImportError: Si `imageio` n'est pas installé pour les modes
                `"video"` ou `"gif"`.
        """
        if save_mode == "human":
            return {}

        elif save_mode == "video":
            try:
                import imageio
            except ImportError:
                raise ImportError("pip install imageio imageio-ffmpeg")
            return {
                "render_mode": "rgb_array",
                "video_dir": output_path + ".mp4",
            }

        elif save_mode == "gif":
            try:
                import imageio
            except ImportError:
                raise ImportError("pip install imageio")
            return {
                "render_mode": "rgb_array",
                "gif_dir": output_path + ".gif",
            }

        else:
            raise ValueError(f"save_mode inconnu : '{save_mode}'. Choix : 'human', 'video', 'gif'")