import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_infer_and_prefix_add_exactly_one_prompt_batch(monkeypatch):
    seen = []

    class Model:
        def sample_actions(self, rng, observation, **kwargs):
            seen.append(observation.tokenized_prompt.shape)
            return np.zeros((1, 2, 3))

        def get_prefix_rep(self, observation):
            seen.append(observation.tokenized_prompt.shape)
            return observation.tokenized_prompt

    class Tokenizer:
        def tokenize(self, prompt, state=None):
            return np.arange(4), np.ones(4, dtype=bool)

    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda function: function)
    policy = _policy.Policy(Model(), transforms=[transforms.TokenizePrompt(Tokenizer())])
    observation = {"state": np.zeros(3), "image": {}, "image_mask": {}, "prompt": "Pick the mug."}
    assert policy.infer(observation)["actions"].shape == (2, 3)
    assert policy.get_prefix_rep(observation).shape == (1, 4)
    assert seen == [(1, 4), (1, 4)]
    for prompts in (["mug"], ["mug", "banana"]):
        for method in (policy.infer, policy.get_prefix_rep):
            with pytest.raises(ValueError, match="requires an unbatched tokenized_prompt"):
                method({**observation, "prompt": prompts})
