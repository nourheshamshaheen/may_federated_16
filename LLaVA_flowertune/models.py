from dataclasses import dataclass, field
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training
from LLaVA.llava.model.language_model.llava_llama import *
import math



class DictToObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig, data_cfg: DictConfig, tokenizer):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    ### NOUR COMMENTED HERE
    # if model_cfg.quantization == 4:
    #     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # elif model_cfg.quantization == 8:
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # else:
    #     raise ValueError(
    #         f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
    #     )

    # # model = AutoModelForCausalLM.from_pretrained(
    # #     model_cfg.name,
    # #     quantization_config=quantization_config,
    # #     torch_dtype=torch.bfloat16,
    # # )
    # model = LlavaForConditionalGeneration.from_pretrained(model_cfg.name,
    #                                                   quantization_config=quantization_config,
    #                                                   torch_dtype=torch.float16)

    # model = prepare_model_for_kbit_training(
    #     model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    # )

    # # peft_config = LoraConfig(
    # #     r=model_cfg.lora.peft_lora_r,
    # #     lora_alpha=model_cfg.lora.peft_lora_alpha,
    # #     lora_dropout=0.075,
    # #     task_type="CAUSAL_LM",
    # # )
    # import re
    # pattern = r'\((\w+)\): Linear'
    # linear_layers = re.findall(pattern, str(model.modules))
    # target_modules = list(set(linear_layers))
    # peft_config = LoraConfig(
    #     r=64,
    #     lora_alpha=16,
    #     target_modules=target_modules
    # )

    # return get_peft_model(model, peft_config)

    ### NOUR STARTED WRITING MODEL HERE
    compute_dtype = (torch.float16) # if error byedrab here change to torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bnb_model_from_pretrained_args = {}
    if model_cfg["quantization"] in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": device},
            # load_in_4bit=model_cfg["quantization"] == 4,
            # load_in_8bit=model_cfg["quantization"] == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=model_cfg["quantization"] == 4,
                load_in_8bit=model_cfg["quantization"] == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))


    model = LlavaLlamaForCausalLM.from_pretrained(
                model_cfg["name"], #model_args.model_name_or_path,
                cache_dir=None,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.float16),
                **bnb_model_from_pretrained_args
            )
    
    model.config.use_cache = False

    # ADD ARGUMENT FREEZE BACKBONE
    if model_cfg["freeze_backbone"]:
        model.model.requires_grad_(False)

    if model_cfg["quantization"] in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32) ## IF ERROR HERE, TORCH.BFLOAT16
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model_cfg["gradient_checkpointing"])

    if model_cfg["gradient_checkpointing"]:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_cfg["lora_enable"]:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=model_cfg["peft_lora_r"],
            lora_alpha=model_cfg["peft_lora_alpha"],
            target_modules=find_all_linear_names(model),
            lora_dropout=model_cfg["lora_dropout"],
            bias=model_cfg["lora_bias"],
            task_type="CAUSAL_LM",
        )
        if model_cfg["quantization"] == 16:
            try:
                model.to(torch.float16)
            except:
                model.to(torch.bfloat16)


        # rank0_print("Adding LoRA adapters...")
        model =  get_peft_model(model, lora_config)

    model_cfg["mm_vision_select_layer"] = -2
    model_cfg["mm_vision_select_feature"] = "patch"
    model_cfg["pretrain_mm_mlp_adapter"] = None
    model_cfg["mm_patch_merge_type"] = "flat"
    model_cfg["mm_use_im_patch_token"] = False
    model_cfg["mm_use_im_start_end"] = False
    model_cfg["mm_projector_type"] = 'linear'
    model_cfg["freeze_mm_mlp_adapter"] = False
    ### ADD VISION TOWER CODE HERE (~30 LINES HERE)
    model.get_model().initialize_vision_modules(
        model_args=DictToObject(**model_cfg),
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float16, device=device)

    data_cfg["image_processor"] = vision_tower.image_processor

    model.config.image_aspect_ratio = data_cfg["image_aspect_ratio"]
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = model_cfg["tune_mm_mlp_adapter"]
    if model_cfg["tune_mm_mlp_adapter"]:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = model_cfg["freeze_mm_mlp_adapter"]
    if model_cfg["freeze_mm_mlp_adapter"]:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if model_cfg["quantization"] in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=device)

    model.config.mm_use_im_start_end = data_cfg["mm_use_im_start_end"] = model_cfg["mm_use_im_start_end"]
    model.config.mm_projector_lr = model_cfg["mm_projector_lr"]
    model.config.mm_use_im_patch_token = model_cfg["mm_use_im_patch_token"]
    model.initialize_vision_tokenizer(DictToObject(**model_cfg), tokenizer=tokenizer)

    if model_cfg["quantization"] in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.float16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if module.weight.dtype == torch.float32:
                        module = module.to(torch.float16)
    
    
    return model




def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
