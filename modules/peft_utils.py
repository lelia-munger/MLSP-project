from peft import LoraConfig, get_peft_model


def add_peft_to_model(model):
    peft_config = LoraConfig(r=32,
                             lora_alpha=64,
                             target_modules=['q_proj', 'v_proj'],
                             bias='none',
                             use_dora=True)
    return get_peft_model(model, peft_config)
