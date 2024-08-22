from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


# before proposed change
MISTRAL_MAPPINGS = [
    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
]
modifier = SmoothQuantModifier(mappings=MISTRAL_MAPPINGS)



# after proposed change
modifier = SmoothQuantModifier()

