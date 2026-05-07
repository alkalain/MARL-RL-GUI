from mario.algos.base import ArchitectureSupport
"""
This is a Wraper which goal is to transforme a MARLlib archtecture in to Mario object

For each class you need to :
- define a default configuration
- the configuration with marllib with the architecture and 
all other hyperparameters the user needs to configurate for the architecture
"""
class MLPArchitecture(ArchitectureSupport):
    """Réseau fully-connected — adapté aux observations vectorielles (MPE, etc.)."""
    def __init__(self, layers: str = "128-128"):
        super().__init__(arch_type="MLP")
        self.layers = layers

    def to_marllib_config(self) -> dict:
        return {
            "core_arch": "mlp",
            "encode_layer": self.layers
        }
