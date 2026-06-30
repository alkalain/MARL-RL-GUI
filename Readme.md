# MARIO - Multi-Agent RL Integration & Optimization

## Installation
L'environnement de développement utilisé pour l'installation est Linux (Ubuntu/Debian). (pour l'instant on ne prévois pas Windows)

### 1. Prérequis :
- Python 3.8 (pour la compatibilité avec MARLlib)
- pip
- virtualenv

### 2. Configuration de l'environnement virtuel :
Afin d'installer l'environnement virtuel nécessaire pour le développement, le script setup_env.py a été mis en place.

En étant à la racine du projet ( `MARL-RL-GUI/` ), exéctuez le script :
```bash
./setup_env.sh
```

### 4. Documentation via Pdoc :
Si vous avez exécuté le script d'installation de l'environnement virtuel, vous pourrez utiliser la documentation via Pdoc 
en étant simplement dans votre environnement virtuel :

Si toutefois vous souhaitez l'installer manuellement, la commande est la suivante :
Installation de pdoc (dans le venv)
```bash
pip install pdoc
```

Pour compiler la documentation : (en étant placé dans le répertoire Mario/)
```bash
pdoc ./Dev/src/mario
```

## Démarrage rapide

Voici un exemple minimal pour entraîner et tester un agent avec MARIO. Ce script configure un environnement multi-agents (via PettingZoo), un algorithme (PPO) et lance une session d'entraînement courte à l'aide du `RunEngine`.

Placez-vous dans `Mario/Dev/src/` avant d'exécuter ce script (ou ajustez le `sys.path` en conséquence) :

```python
from mario.engine import RunEngine
from mario.algos.ppo import PPOAlgo
from mario.algos.architectures import MLPArchitecture
from mario.envs.pettingzoo_env import PettingZooEnvWrapper

# 1. Choix de l'environnement (ici : Multi-Agent Particle Environments)
env = PettingZooEnvWrapper(env_name="mpe", map_name="simple_world_comm")

# 2. Choix de l'architecture réseau et de l'algorithme
architecture = MLPArchitecture(layers="64-64")
algo = PPOAlgo(architecture=architecture)

# 3. Lancement de l'entraînement via le moteur MARIO
engine = RunEngine()
policy = engine.run_training(
    env=env,
    algorithme=algo,
    architecture=architecture,
    stop_criteria={"training_iteration": 3},  # entraînement court, à but de démonstration
)

# 4. Visualisation de la politique entraînée
engine.run_render(policy, env, save_mode="human")
```

Pour aller plus loin, un script interactif complet (sélection de l'environnement, de l'algorithme PPO/QMIX, et rendu) est disponible dans :
```bash
python Mario/Dev/src/tests/main_test.py
```

> ℹ️ L'entraînement s'appuie sur [MARLlib](https://marllib.readthedocs.io/) et [Ray](https://www.ray.io/) en interne : un premier lancement peut prendre un peu de temps le temps que les dépendances s'initialisent.
