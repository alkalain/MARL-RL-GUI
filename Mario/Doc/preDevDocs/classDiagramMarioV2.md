```mermaid
classDiagram
    %% --- MOTEUR PRINCIPAL ---
    class RunEngine {
        +String type
        +run_training(env, algo, arch, algo_hpo, archi_hpo) JointPolicy
        +run_test(policy, env) Stats
    }

    %% --- ENVIRONNEMENTS ---
    class MultiAgentEnv {
        <<abstract>>
        +String type
        +reset()*
        +step(actions)*
    }
    class ParallelEnv {
        <<abstract>>
    }
    class PettingZooEnvWrapper {
        -env pz_env
        +reset()
        +step(actions)
        +close()
    }
    MultiAgentEnv <|-- ParallelEnv
    ParallelEnv <|-- PettingZooEnvWrapper

    %% --- ALGORITHMES & POLITIQUES ---
    class Algo {
        <<abstract>>
        +String type
    }
    class PPOAlgo {
        +dict hyperparams
        +train(env, total_timesteps) PPOPolicy
    }
    Algo <|-- PPOAlgo

    class JointPolicy {
        <<abstract>>
        +String type
        +predict(observations)*
    }
    class PPOPolicy {
        -model
        +predict(observations)
    }
    JointPolicy <|-- PPOPolicy

    class ArchitectureSupport {
        <<abstract>>
        +String type
    }

    %% --- ESPACES DE RECHERCHE (HPO) ---
    class AlgoHyperparametersResearchSpace {
        <<abstract>>
        +String type
    }
    class ArchiHyperparametersResearchSpace {
        <<abstract>>
        +String type
    }

    %% --- DÉPENDANCES DU MOTEUR ---
    RunEngine ..> MultiAgentEnv : utilise
    RunEngine ..> Algo : utilise
    RunEngine ..> ArchitectureSupport : utilise
    RunEngine ..> AlgoHyperparametersResearchSpace : utilise
    RunEngine ..> ArchiHyperparametersResearchSpace : utilise
    RunEngine ..> JointPolicy : génère (via train)
    ```