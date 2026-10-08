import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert data["tokenized_prompt"].shape == (12,)
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


class _DummyTokenizer:
    def tokenize(self, prompt, state=None, actions=None):
        tokens = np.arange(4)
        mask = np.ones(4, dtype=bool)
        return (tokens, mask, mask, mask) if actions is not None else (tokens, mask)


def test_resize_images_is_applied_before_batching():
    transform = _transforms.ResizeImages(224, 224)
    single = transform({"image": {"base": np.zeros((256, 128, 3), dtype=np.uint8)}})
    batch = transform({"image": {"base": np.zeros((2, 256, 128, 3), dtype=np.uint8)}})
    assert single["image"]["base"].shape == (224, 224, 3)
    assert batch["image"]["base"].shape == (2, 224, 224, 3)


@pytest.mark.parametrize(
    ("prompt", "shape"),
    [
        ("mug", (4,)),
        (np.asarray(b"mug"), (4,)),
        (["mug"], (1, 4)),
        (np.asarray(["mug", "banana"]), (2, 4)),
    ],
)
def test_prompt_shape_preserves_input_batch_contract(prompt, shape):
    output = _transforms.TokenizePrompt(_DummyTokenizer())({"prompt": prompt})
    assert output["tokenized_prompt"].shape == shape
    assert output["tokenized_prompt_mask"].shape == shape


@pytest.mark.parametrize("batched", [False, True])
def test_fast_prompt_shape_preserves_input_batch_contract(batched):
    data = {"prompt": ["mug"] if batched else "mug", "state": np.zeros(3), "actions": np.zeros((2, 3))}
    output = _transforms.TokenizeFASTInputs(_DummyTokenizer())(data)
    for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
        assert output[key].shape == ((1, 4) if batched else (4,))


def test_fast_mismatched_prompt_and_action_batches_fail():
    data = {"prompt": ["mug"], "state": np.zeros((1, 3)), "actions": np.zeros((2, 2, 3))}
    with pytest.raises(ValueError, match="Batched actions and prompt lengths"):
        _transforms.TokenizeFASTInputs(_DummyTokenizer())(data)
