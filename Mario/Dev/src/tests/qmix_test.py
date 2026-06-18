from marllib import marl

# QMIX nécessite un environnement coopératif avec actions discrètes
env = marl.make_env(
    environment_name="mpe",
    map_name="simple_spread",
    force_coop=True,
    continuous_actions=False,
)

# "test" = rapide/debug ; utilise "common" ou "mpe" pour un entraînement plus sérieux
qmix = marl.algos.qmix(hyperparam_source="test")

# QMIX utilise typiquement un modèle récurrent
model = marl.build_model(
    env,
    qmix,
    {
        "core_arch": "gru",
        "encode_layer": "128-256",
    },
)

qmix.fit(
    env,
    model,
    stop={"training_iteration": 10},
    local_mode=True,
    num_gpus=0,
    num_workers=2,
    share_policy="all",
    checkpoint_end=True,
)