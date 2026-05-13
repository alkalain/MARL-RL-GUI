# MARIO - Multi-Agent RL Integration & Optimization

## Installation
L'environnement de développement utilisé pour l'installation est Linux (Ubuntu/Debian). (pour l'instant on ne prévois pas Windows)

### 1. Prérequis :
- Python 3.8 (pour la compatibilité avec MARLlib)
- pip
- virtualenv

### 2. Configuration de l'environnement virtuel :
Pour créer l'environnement virtuel, il est préférable d'être dans le répertoire Mario/
```bash
cd Mario/
```

Création de l'environnement virtuel :
```bash
python3 -m venv venv38
```

Pour activer l'environnement (se place au préalable dans son répertoire)
```bash
source venv38/bin/activate
```

Installation de pip pour gérer les installations futures :
```bash
python -m pip install --upgrade pip
```


### 3. Installation des dépendances :
Installation des dépendances de base :
```bash
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install "protobuf<3.20"
pip install gym==0.21.0
pip install "numpy==1.23.5"
```

Installation du package pour tester le setup.py :
```bash
pip install -e Mario/Dev/src/
```

Installation minimales de dépendances afin que les imports ne plantent pas
```bash
pip install marllib pettingzoo[mpe]
```

### 4. Documentation via Pdoc :
Installation de pdoc (dans le venv)
```bash
pip install pdoc
```

Pour compiler la documentation : (en étant placé dans le répertoire Mario/)
```bash
pdoc ./Dev/src/mario
```