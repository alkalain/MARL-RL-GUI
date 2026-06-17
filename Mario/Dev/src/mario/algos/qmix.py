from marllib import marl
from mario.algos.base import Algo, ArchitectureSupport
from mario.algos.architectures import MLPArchitecture

class QMixAlgo(Algo):
    """
    Implémentation de l'algorithme Q-Mix s'appuyant sur MARLlib.
    """
    def __init__(
            self,
            architecture: ArchitectureSupport = None,
            hyperparams: dict = None,
            ):
        super().__init__(
            algo_type="QMix",
            hyperparams=hyperparams or {
                "lr": 0.0005,
                "batch_size": 64,
            },
            share_policy="all"
        )
        self.architecture = architecture or MLPArchitecture()

    def _get_marllib_algo(self, env_name: str):
        return marl.algos.qmix(hyperparam_source="test")
    