#!/bin/bash

# Arrêter le script en cas d'erreur
set -e

cd Mario/

echo "=== 1. Création de l'environnement virtuel venv38 ==="
python3.8 -m venv venv38

echo "=== 2. Activation de l'environnement virtuel ==="
source venv38/bin/activate

echo "=== 3. Mise à jour de pip ==="
python -m pip install --upgrade pip

echo "=== 4. Installation des dépendances de base ==="
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install "protobuf<3.20"
pip install gym==0.21.0
pip install "numpy==1.23.5"

echo "=== 5. Installation des dépendances via le fichier requirements.txt ==="
if [ -f "../requirements.txt" ]; then
    pip install -r ../requirements.txt
else
    echo "Attention : ../requirements.txt introuvable, étape sautée."
fi

echo "=== 6. Installation du package local (mode éditable) ==="
if [ -d "Dev/src/" ]; then
    pip install -e Dev/src/
else
    echo "Attention : Le dossier Dev/src/ introuvable, étape sautée."
fi

echo "=== 7. Installation de marllib, pettingzoo et pdoc ==="
pip install marllib pettingzoo[mpe]
pip install pdoc

echo "======== Installation terminée avec succès ! ========"
echo "Pour activer l'environnement dans votre terminal actuel, placez vous d'abord dans le répertoire Mario/ :"
echo "cd Mario/"

echo "Puis lancez la commande d'activation de l'environnement :"
echo "source Mario/venv38/bin/activate"