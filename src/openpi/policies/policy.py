from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _validate_unbatched_prompt(inputs: dict) -> dict:
    """Policy methods accept one observation; never discard additional prompts."""
    for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
        if key in inputs and np.ndim(inputs[key]) != 1:
            raise ValueError(f"Policy requires an unbatched {key} with shape (L,), got {np.shape(inputs[key])}")
    return inputs


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        seed: int = 0,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            seed: Seed for this policy's private action-sampling stream.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._torch_generator = torch.Generator(device=pytorch_device)
            self._torch_generator.manual_seed(int(seed))
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng if rng is not None else jax.random.key(int(seed))
            if hasattr(model, "get_prefix_rep"):
                self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        return_vlm_embedding: bool = False,
        return_normalized: bool = False,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = _validate_unbatched_prompt(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if self._is_pytorch_model:
            sample_kwargs.setdefault("generator", self._torch_generator)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        if not return_normalized:
            outputs = self.apply_output_transforms(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        if return_vlm_embedding:
            vlm_embedding = self._get_prefix_rep(_model.Observation.from_dict(inputs))
            outputs["vlm_embedding"] = vlm_embedding

        return outputs

    def seed(self, seed: int) -> None:
        """Reset this policy's private action-sampling stream."""

        if self._is_pytorch_model:
            self._torch_generator.manual_seed(int(seed))
        else:
            self._rng = jax.random.key(int(seed))

    def get_rng_state(self) -> dict:
        """Return a host-resident RNG snapshot suitable for checkpointing."""

        if self._is_pytorch_model:
            return {
                "framework": "torch",
                "state": self._torch_generator.get_state().cpu().numpy(),
            }
        return {
            "framework": "jax",
            "key_data": np.asarray(jax.random.key_data(self._rng)),
            "implementation": str(jax.random.key_impl(self._rng)),
        }

    def set_rng_state(self, state: dict) -> None:
        """Restore a state returned by :meth:`get_rng_state`."""

        framework = state.get("framework")
        if framework == "torch" and self._is_pytorch_model:
            # Network/msgpack decoders may return a read-only NumPy view.  Torch
            # warns that wrapping it directly permits undefined writes, so own
            # the small generator-state buffer before restoring it.
            value = torch.tensor(np.array(state["state"], copy=True), dtype=torch.uint8, device="cpu")
            self._torch_generator.set_state(value)
            return
        if framework == "jax" and not self._is_pytorch_model:
            self._rng = jax.random.wrap_key_data(
                jnp.asarray(state["key_data"], dtype=jnp.uint32),
                impl=state.get("implementation"),
            )
            return
        raise ValueError(
            f"Cannot restore {framework!r} RNG state into "
            f"{'PyTorch' if self._is_pytorch_model else 'JAX'} policy"
        )

    def apply_output_transforms(self, outputs: dict) -> dict:
        """Apply configured denormalization/robot transforms to normalized output."""

        return self._output_transform(outputs)

    def get_prefix_rep(self, obs: dict):
        """Returns (hidden_state, kv_cache) from the VLM prefix pass, without generating actions."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = _validate_unbatched_prompt(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return self._get_prefix_rep(_model.Observation.from_dict(inputs))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
