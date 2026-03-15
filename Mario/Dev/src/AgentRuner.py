# AgentRuner est le fichier qui vas permetre de lancer l'agent
from gymnasium.spaces import Discrete, MultiDiscrete

class Algorithm:

    def __init__(self, name):
        self.name = name

    def __init__(self, name,HyperparametersResearcherSpace):
        self.name = name
        self.HyperparametersResearcherSpace = HyperparametersResearcherSpace


class ArchitectureSupport:

    def __init__(self, name):
        self.name = name

    def __init__(self, name,HyperparametersResearcherSpace):
        self.name = name
        self.HyperparametersResearcherSpace = HyperparametersResearcherSpace

class HyperparametersResearcherSpace:

    def __init__(self, name):
        self.name = name

    def __init__(self, name,ObsSpaceType,ActionSpaceType):
        self.name = name
        self.ObsSpaceType = ObsSpaceType #
        self.ActionSpaceType = ActionSpaceType