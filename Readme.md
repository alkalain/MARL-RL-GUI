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