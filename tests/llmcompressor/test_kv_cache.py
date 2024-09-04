import os
import tempfile

import pytest
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization.cache import KVCacheScaleType
from compressed_tensors.quantization.utils.helpers import iter_named_quantizable_modules
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

MODEL_IDS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "mistralai/Mistral-7B-v0.1" "deepseek-ai/DeepSeek-Coder-V2-Instruct",
]


@pytest.fixture(scope="session")
def oneshot_fixture():
    num_bits = 8
    _type = "float"
    strategy = "tensor"
    dynamic = False
    symmetric = True

    recipe = f"""
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: {num_bits}
                    type: {_type}
                    strategy: {strategy}
                    dynamic: {dynamic}
                    symmetric: {symmetric}
    """

    used_args = dict(
        num_bits=num_bits,
        _type=_type,
        strategy=strategy,
        dynamic=dynamic,
        symmetric=symmetric,
    )

    oneshot_args = dict(
        dataset="open_platypus",
        recipe=recipe,
        num_calibration_samples=16,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        for model_id in MODEL_IDS:
            oneshot_args["output_dir"] = os.path.join(tmpdir, model_id)
            used_args["output_dir"] = oneshot_args["output_dir"]
            yield oneshot(model=model_id, **oneshot_args), used_args


def test_kv_cache_config_format(oneshot_fixture):
    _, used_args = oneshot_fixture

    output_dir = used_args["output_dir"]

    compressor = ModelCompressor.from_pretrained(output_dir)
    # check config is properly populated
    quant_config = compressor.quantization_config
    assert quant_config is not None
    assert quant_config.kv_cache_scheme is not None

    # check the values match
    scheme = quant_config.kv_cache_scheme
    assert scheme.num_bits == used_args["num_bits"]
    assert scheme.type == used_args["_type"]
    assert scheme.strategy == used_args["strategy"]
    assert scheme.dynamic == used_args["dynamic"]
    assert scheme.symmetric == used_args["symmetric"]


def test_kv_cache_model_state_dict_attr(oneshot_fixture):
    _, used_args = oneshot_fixture

    output_dir = used_args["output_dir"]

    model = SparseAutoModelForCausalLM.from_pretrained(output_dir)

    counts = 0
    for name, submodule in iter_named_quantizable_modules(
        model, include_children=False, include_attn=True
    ):
        counts += 1
        assert "self_attn" in name
        assert hasattr(submodule, KVCacheScaleType.VALUE.value)
        assert hasattr(submodule, KVCacheScaleType.KEY.value)

    assert counts > 0


def test_kv_cache_with_gptq(tmp_path):
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = SparseAutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Select calibration dataset.
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    # Select number of samples. 512 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 16
    MAX_SEQUENCE_LENGTH = 512
    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize inputs.
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    kv_cache_num_bits = 8
    kv_cache_type = "float"
    kv_cache_strategy = "tensor"
    kv_cache_dynamic = False
    kv_cache_symmetric = True

    recipe = f"""
    quant_stage:
        quant_modifiers:
            GPTQModifier:
                sequential_update: false
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 4
                            type: "int"
                            symmetric: true
                            strategy: "channel"
                            actorder: False
                        targets: ["Linear"]
        QuantizationModifier:
            kv_cache_scheme:
                num_bits: {kv_cache_num_bits}
                type: {kv_cache_type}
                strategy: {kv_cache_strategy}
                dynamic: {kv_cache_dynamic}
                symmetric: {kv_cache_symmetric}
    """
    output_dir = str(tmp_path)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        output_dir=output_dir,
    )

    compressor = ModelCompressor.from_pretrained(output_dir)

    # check config is properly populated
    quant_config = compressor.quantization_config
    assert quant_config is not None
    assert quant_config.kv_cache_scheme is not None

    # check the values match
    scheme = quant_config.kv_cache_scheme
    assert scheme.num_bits == kv_cache_num_bits
    assert scheme.type == kv_cache_type
    assert scheme.strategy == kv_cache_strategy
    assert scheme.dynamic == kv_cache_dynamic
    assert scheme.symmetric == kv_cache_symmetric

    model = SparseAutoModelForCausalLM.from_pretrained(output_dir)

    counts = 0
    for name, submodule in iter_named_quantizable_modules(
        model, include_children=False, include_attn=True
    ):
        counts += 1
        assert "self_attn" in name
        assert hasattr(submodule, KVCacheScaleType.VALUE.value)
        assert hasattr(submodule, KVCacheScaleType.KEY.value)

    assert counts > 0
