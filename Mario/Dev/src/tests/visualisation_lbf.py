import gym
import lbforaging
import json
import os
import torch
# Assure-toi d'avoir importé ta classe de politique si besoin
# from mario.algos.marllibpolicy import MARLlibPolicy 

def visualiser_run(exp_path):
    # 1. Retrouver les paramètres de l'environnement (sauvegardés par MARLlib)
    params_path = os.path.join(exp_path, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    env_name = params["env_config"]["env"]
    map_name = params["env_config"]["map_name"]
    
    print(f"Lancement de la simulation pour : {map_name}")
    
    # 2. Création de l'environnement natif (sans Ray/MARLlib)
    env = gym.make(map_name)
    
    # 3. Boucle de rendu
    obs = env.reset()
    done = False
    
    while not done:
        # Ici, dans un vrai cas, tu chargerais ta politique torch pour prédire l'action
        # Pour l'instant, on laisse l'agent faire des actions aléatoires pour tester le rendu
        actions = env.action_space.sample() 
        
        obs, reward, done, info = env.step(actions)
        
        # Le rendu natif Pygame
        env.render()
    
    env.close()
    print("Simulation terminée.")

if __name__ == "__main__":
    # Remplace par ton chemin réel
    chemin = "/home/vignee/MIASHS1/TER/MARL-RL-GUI/Mario/Dev/src/exp_results/mappo_mlp_Foraging-8x8-2p-2f-v2/MAPPOTrainer_lbf_Foraging-8x8-2p-2f-v2_4cadb_00000_0_2026-06-06_19-51-09"
    visualiser_run(chemin)