import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.ablation_robustness import controlled_action, factorial_interactions
from sacsi_full import SACSIConfig, SACSIRecurrentAgent


def test_full_controlled_action_matches_actor_and_no_context_matches_base():
    torch.manual_seed(11)
    agent = SACSIRecurrentAgent(SACSIConfig(), device="cpu")
    state = (
        np.zeros(8, np.float32),
        np.zeros((24, 8), np.float32),
        np.zeros(3, np.float32),
    )
    direct = float(agent.select_action(state, deterministic=True)[0])
    assert controlled_action(agent.actor, state, "Full") == pytest.approx(direct, abs=1e-7)

    current = torch.as_tensor(state[0]).unsqueeze(0)
    with torch.inference_mode():
        hidden = agent.actor.base.body(current)
        expected = float(((torch.tanh(agent.actor.base.mean(hidden)) + 1) * 2.5)[0, 0])
    assert controlled_action(agent.actor, state, "No Context") == pytest.approx(expected, abs=1e-7)


def test_factorial_interaction_is_matched_by_seed():
    rows = []
    values = {"SAC Basic": 1, "SAC + Forecast": 2, "SAC + LSTM": 3, "SACSI Full": 7}
    for seed in (11, 22):
        for method, value in values.items():
            rows.append({
                "seed": seed, "method": method, "method_type": "rl", "metric": value + seed,
            })
    result = factorial_interactions(pd.DataFrame(rows), ("metric",))
    assert result["metric_interaction"].tolist() == [3.0, 3.0]
