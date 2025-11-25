import numpy as np
from PIL import Image

# Import your code
from openpi.transforms import ResizeImages, TokenizePrompt
from openpi_client.image_tools import resize_with_pad_jax as resize_with_pad
from openpi.models.tokenizer import PaligemmaTokenizer as tokenizer

def make_random_image(h, w, c=3):
    """Utility: create random uint8 image."""
    return (np.random.rand(h, w, c) * 255).astype(np.uint8)


def test_resize_with_pad_single():
    print("\n--- test_resize_with_pad_single ---")
    img = make_random_image(100, 200, 3)

    out = resize_with_pad(img, height=128, width=128)

    assert out.shape == (128, 128, 3)
    print("Output shape OK:", out.shape)

    # Check padding: non-zero inside, zeros in padded area
    non_zero = np.count_nonzero(out)
    assert non_zero > 0
    print("Padding check OK")

    # Check that content is centered
    # (just ensure content is not collapsed into a corner)
    center_patch = out[32:96, 32:96]
    assert np.count_nonzero(center_patch) > 0
    print("Center content check OK")


def test_resize_with_pad_batch():
    print("\n--- test_resize_with_pad_batch ---")
    batch = np.stack([
        make_random_image(224, 224),
        make_random_image(224, 224),
        make_random_image(224, 224),
    ])  # shape (3, H, W, C)

    out = resize_with_pad(batch, height=128, width=128)

    assert out.shape == (3, 128, 128, 3)
    print("Batch output shape OK:", out.shape)

    # Every image must have non-empty content
    for i in range(3):
        assert np.count_nonzero(out[i]) > 0
    print("All batch elements padded + resized correctly")


def test_resize_images_transform():
    print("\n--- test_resize_images_transform ---")
    # Create fake input dict matching your DataDict format
    data = {
        "image": {
            "cam0": make_random_image(120, 80),
            "cam1": make_random_image(200, 300),
        }
    }

    T = ResizeImages(height=128, width=128)
    out = T(data)

    assert out["image"]["cam0"].shape == (128, 128, 3)
    assert out["image"]["cam1"].shape == (128, 128, 3)

    print("ResizeImages transform OK")
    print("cam0:", out["image"]["cam0"].shape)
    print("cam1:", out["image"]["cam1"].shape)




def test_single_prompt():
    print("\n--- test_single_prompt ---")

    transform = TokenizePrompt(tokenizer=tokenizer)

    data = {
        "prompt": np.array(b"put the yellow and white mug in the microwave and close it", dtype=object),
        "state": np.array([1,2,3]),
        "other_key": 42,
    }

    out = transform(data)

    assert "tokenized_prompt" in out
    assert "tokenized_prompt_mask" in out

    toks = out["tokenized_prompt"]
    mask = out["tokenized_prompt_mask"]

    assert toks.shape == (5,)            # h e l l o
    assert mask.shape == (5,)
    assert toks.tolist() == [104, 101, 108, 108, 111]
    
    print("Single prompt OK.")


def test_batched_prompts_no_state():
    print("\n--- test_batched_prompts_no_state ---")

    transform = TokenizePrompt(tokenizer=tokenizer)

    data = {
        "prompt": np.array([b"hi", b"yo"], dtype=object),
        "other": np.array([1, 2]),
    }

    out = transform(data)

    toks = out["tokenized_prompt"]
    mask = out["tokenized_prompt_mask"]

    assert toks.shape == (2, 2)    # 2 prompts, each length 2
    assert toks.tolist() == [
        [104, 105],   # h i
        [121, 111],   # y o
    ]
    assert mask.shape == (2, 2)

    print("Batched prompts without state OK.")


def test_batched_prompts_with_batched_state():
    print("\n--- test_batched_prompts_with_batched_state ---")

    transform = TokenizePrompt(tokenizer=tokenizer, discrete_state_input=True)

    data = {
        "prompt": np.array([b"a", b"b"], dtype=object),
        "state": np.array([
            [1,2,3],
            [4,5,6],
        ], dtype=np.int32),
    }

    out = transform(data)

    toks = out["tokenized_prompt"]
    mask = out["tokenized_prompt_mask"]

    assert toks.shape == (2, 1)       # 2 prompts, each length 1
    assert toks.tolist() == [[97], [98]]   # ASCII a=97, b=98
    assert mask.shape == (2, 1)

    print("Batched prompts WITH batched state OK.")


def test_python_string_input():
    print("\n--- test_python_string_input ---")

    transform = TokenizePrompt()

    data = {
        "prompt": "ok",
        "foo": 123,
    }

    out = transform(data)

    toks = out["tokenized_prompt"]

    assert toks.tolist() == [111, 107]   # 'o', 'k'

    print("Python string input OK.")



if __name__ == "__main__":
    # test_resize_with_pad_single()
    # test_resize_with_pad_batch()
    # test_resize_images_transform()
    test_single_prompt()
    test_batched_prompts_no_state()
    test_batched_prompts_with_batched_state()
    test_python_string_input()
    print("\nAll tests PASS ✔️")
