import os
import glob
from pathlib import Path
from mario.algos.base import JointPolicy

class MARLlibPolicy(JointPolicy):
    """
    Wrapper pour transformer le résultat de MARLlib en une 
    politique compatible avec l'interface MARIO.
    """
    def __init__(self, model, algo_instance, env_instance,exp_pattern):
        super().__init__(policy_type="MARLlib_Policy")
        self.model = model
        self.algo = algo_instance
        self.env = env_instance
        self.exp_pattern = exp_pattern

    def predict(self, observations):
        """
        À implémenter plus tard : utilisera self.algo.render ou 
        self.model pour calculer les actions.
        """
        return {}

    def render(self,env=None,model=None,save_mode: str = "human"):
        """Utilise env et model stockés si non fournis."""
        env = env or self.env
        model = model or self.model

        script_dir = Path(__file__).resolve().parent
        exp_results_dir = script_dir.parent.parent / "exp_results"

        pattern = str(exp_results_dir / self.exp_pattern)
        run_dirs = glob.glob(pattern)

        if not run_dirs:
            raise FileNotFoundError(
                f"Aucun résultat trouvé.\nPattern : {pattern}"
            )

        base_path = max(run_dirs, key=os.path.getctime)
        print(f"[MARIO] Run sélectionné  : {base_path}")

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