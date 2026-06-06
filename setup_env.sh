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

echo "=== 6. Installation spécifique (lbforaging compatible) ==="
pip install "lbforaging==1.1.1" --no-deps

echo "=== 7. Configuration MARLlib (Patch mappo.yaml) ==="
VENV_PATH="venv38/lib/python3.8/site-packages"
CONFIG_DIR="$VENV_PATH/examples/config/algo_config"
mkdir -p "$CONFIG_DIR"
cat <<EOF > "$CONFIG_DIR/mappo.yaml"
algo_args:
  use_gae: True
  lambda: 0.95
  gamma: 0.99
  clip_param: 0.2
  ppo_epoch: 10
  num_mini_batch: 1
  value_loss_coef: 1.0
  entropy_coef: 0.01
  lr: 0.0005
  critic_lr: 0.0005
  clip_range: 0.2
  vf_clip_param: 10.0
  kl_target: 0.016
  kl_coeff: 0.2
  batch_episode: 10
  batch_mode: complete_episodes
  use_critic: True
  local_dir: ""
  checkpoint_freq: 10
  interv_steps: 100
  num_sgd_iter: 10
  vf_loss_coeff: 1.0
  entropy_coeff: 0.01
EOF

echo "=== 8. Installation du package local (mode éditable) ==="
if [ -d "Dev/src/" ]; then
    pip install -e Dev/src/
else
    echo "Attention : Le dossier Dev/src/ introuvable, étape sautée."
fi

echo "=== 9. Installation de marllib, pettingzoo et pdoc ==="
pip install marllib pettingzoo[mpe]
pip install pdoc

echo "======== Installation terminée avec succès ! ========"
echo "Pour activer l'environnement dans votre terminal actuel, placez vous d'abord dans le répertoire Mario/ :"
echo "cd Mario/"

echo "Puis lancez la commande d'activation de l'environnement :"
echo "source venv38/bin/activate"