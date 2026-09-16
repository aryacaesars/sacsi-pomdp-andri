"""Central experiment registry for SACSI-POMDP Research Release."""

EXPERIMENTS = {
    "main_benchmark": {
        "name": "Fair DRL Benchmark",
        "methods": ["DDPG", "TD3", "SAC", "SACSI"],
    },
    "pomdp_ablation": {
        "name": "POMDP Component Ablation",
        "methods": ["SAC", "SAC_Forecast", "SAC_LSTM", "SACSI"],
    },
}

def get_experiment(name):
    return EXPERIMENTS[name]
