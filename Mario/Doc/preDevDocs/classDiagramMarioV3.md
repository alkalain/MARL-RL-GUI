```mermaid
classDiagram
    %% Core Engine & Stats
    class RunEngine {
        +str type
        +run_training(env, algorithme, architecture, algo_hpo_space, archi_hpo_space, ...) MARLlibPolicy
        +run_render(policy, env, save_mode) bool
    }

    %% Environments
    class MultiAgentEnv {
        <<abstract>>
        +str type
        +reset()*
        +step(actions)*
    }
    class ParallelEnv {
        <<abstract>>
    }
    class PettingZooEnvWrapper {
        +str env_name
        +str map_name
        +dict env_kwargs
        +Any env
        +reset()
        +step(actions)
        +close()
    }

    MultiAgentEnv <|-- ParallelEnv
    ParallelEnv <|-- PettingZooEnvWrapper
    RunEngine ..> MultiAgentEnv : utilise

    %% Architectures
    class ArchitectureSupport {
        <<abstract>>
        +str type
        +to_marllib_config()* dict
    }
    class MLPArchitecture {
        +str layers
        +to_marllib_config() dict
    }
    class GRUArchitecture {
        +str layers
        +int hidden_state_size
        +to_marllib_config() dict
    }
    class CNNArchitecture {
        +str layers
        +to_marllib_config() dict
    }
    class RNNArchitecture {
        +str layers
        +to_marllib_config() dict
    }

    ArchitectureSupport <|-- MLPArchitecture
    ArchitectureSupport <|-- GRUArchitecture
    ArchitectureSupport <|-- CNNArchitecture
    ArchitectureSupport <|-- RNNArchitecture

    %% Algorithms
    class Algo {
        <<abstract>>
        +str type
        +dict hyperparams
        +str share_policy
        +train(env_name, map_name, architecture, stop_criteria, ...) MARLlibPolicy
        #_get_marllib_algo(env_name)*
    }
    class PPOAlgo {
        +ArchitectureSupport architecture
        #_get_marllib_algo(env_name)
    }
    class QMixAlgo {
        +ArchitectureSupport architecture
        #_get_marllib_algo(env_name)
    }

    Algo <|-- PPOAlgo
    Algo <|-- QMixAlgo
    Algo --> ArchitectureSupport : configure
    RunEngine ..> Algo : utilise

    %% Policies
    class JointPolicy {
        <<abstract>>
        +str type
        +predict(observations)* dict
    }
    class MARLlibPolicy {
        +Any algo
        +Any env
        +Any model
        +predict(observations) dict
        +render(env, model, save_mode)
        +_convert_mp4_to_gif(record_dir, output_path)
    }

    JointPolicy <|-- MARLlibPolicy
    Algo ..> MARLlibPolicy : génère en sortie
    RunEngine ..> JointPolicy : évalue/rend

    %% HPO Space & Optimizer
    class AlgoHyperparametersResearchSpace {
        <<abstract>>
        +str type
        +suggest(trial)* dict
    }
    class ArchiHyperparametersResearchSpace {
        <<abstract>>
        +str type
        +suggest(trial)* dict
    }
    class PPOAlgoSpace {
        +suggest(trial) dict
    }
    class MLPArchiSpace {
        +suggest(trial) dict
    }

    AlgoHyperparametersResearchSpace <|-- PPOAlgoSpace
    ArchiHyperparametersResearchSpace <|-- MLPArchiSpace

    class HPOptimizer {
        +type algo_class
        +str env_name
        +str map_name
        +dict env_kwargs
        +AlgoHyperparametersResearchSpace algo_space
        +ArchiHyperparametersResearchSpace archi_space
        +str study_name
        +str storage_url
        +optimize(n_trials, n_workers) Algo
        #_objective(trial) float
        #_run_worker(trial_id) float
    }

    HPOptimizer --> AlgoHyperparametersResearchSpace : interroge
    HPOptimizer --> ArchiHyperparametersResearchSpace : interroge
    HPOptimizer ..> Algo : instancie le meilleur
    RunEngine ..> HPOptimizer : délègue si HPO activé
    ```