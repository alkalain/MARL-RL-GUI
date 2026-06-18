from marllib import marl

# prepare env
env = marl.make_env(environment_name="overcooked",
                    map_name="asymmetric_advantages")

# initialize algorithm with appointed hyper-parameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")
mappo = marl.algos.mappo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
# 
mappo.fit(env, model, stop={'episode_reward_mean': 6000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0, num_gpus_per_worker=0, num_workers=1, share_policy='group', checkpoint_freq=20)

# rendering
mappo.render(env, model,
             restore_path={
                 'params_path': "./exp_results/mappo_mlp_asymmetric_advantages_copy/MAPPOTrainer_overcooked_asymmetric_advantages_08cc3_00000_0_2025-03-19_10-25-01/params.json",
                 'model_path': "./exp_results/mappo_mlp_asymmetric_advantages_copy/MAPPOTrainer_overcooked_asymmetric_advantages_08cc3_00000_0_2025-03-19_10-25-01/checkpoint_000020/checkpoint-20",
                #  'render': True,
                 #  'record_env': True,
                 'render_env': True
             },
             enable_temm = True,
             local_mode=True,
             share_policy="group",
             stop_timesteps=1,
             timesteps_total=1,
             checkpoint_freq=1,
             stop_iters=1,
             checkpoint_end=True)