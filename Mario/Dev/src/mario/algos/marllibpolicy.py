import os
import glob
import json
from marllib import marl
from pathlib import Path
from mario.algos.base import JointPolicy
# note pour plus tard  "pyglet==1.5.27" à ajouter à la doc
# meme chose pour "Pillow==9.5.0"
class MARLlibPolicy(JointPolicy):
    """
    Classe assurant la compatibilité entre les modèles produits par MARLlib
    et l'interface de décision définie par MARIO.

    Cette classe encapsule les composants nécessaires (modèle, algorithme, environnement)
    pour permettre l'execution et le rendu des agents après leur entraînement.
    """
    def __init__(self, model, algo_instance, env_instance,exp_pattern):
        """
        Initialise le conteneur de la politique entraînée.

        Args:
            model: Objet réseau de neurones (Policy/Model) généré par MARLlib.
            algo_instance: Instance de l'algorithme ayant servi à l'apprentissage.
            env_instance: Référence vers l'environnement de simulation associé.
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
    
    def _load_params(self, base_path: Path) -> dict:
        """Charge params.json et retourne le contenu."""
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
        """Cherche exp_results/ en remontant l'arborescence jusqu'à la racine."""
        current = Path(__file__).resolve().parent
        while current != current.parent:
            candidate = current / "exp_results"
            if candidate.exists():
                return candidate
            current = current.parent
        raise FileNotFoundError(
            f"Aucun dossier exp_results/ trouvé depuis {Path(__file__).resolve().parent}"
        )

    def _recreate_env(self, base_path: Path):
        """Recrée l'env MARLlib depuis les infos de params.json."""
        params = self._load_params(base_path)
        env_args = params["model"]["custom_model_config"]["env_args"]
        env_name = params["model"]["custom_model_config"]["env"]
        map_name = env_args["map_name"]

        print(f"[MARIO] Env recréé depuis params.json : {env_name}:{map_name}")
        return marl.make_env(environment_name=env_name, map_name=map_name)

    def render(self,env=None,model=None,save_mode: str = "human"):
        """Utilise env et model stockés si non fournis."""
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

    def _build_render_config(self, save_mode: str, output_path: str) -> dict:
        """
        Construit les paramètres de rendu selon le mode choisi.
        human : rendu à en fenêtre
        video : sauvegarde en .mp4
        gif : sauvegarde en .gif
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