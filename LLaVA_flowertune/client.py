from collections import OrderedDict
from typing import Callable, Dict, Tuple
from LLaVA.llava.train.llava_trainer import LLaVATrainer
import flwr as fl
import torch
import os
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig, OmegaConf
from trl import SFTTrainer
from transformers import TrainingArguments, LlavaForConditionalGeneration
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from dataset import get_data_module
from models import get_model, cosine_annealing
import pathlib
import logging
import transformers

class DictToObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        data_cfg: DictConfig,
        tokenizer,
        # formatting_prompts_func,
        save_path,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.train_cfg = OmegaConf.to_container(self.train_cfg, resolve=True)
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self.train_cfg = OmegaConf.merge(self.train_cfg, data_cfg)
        self.train_cfg = OmegaConf.merge(self.train_cfg, model_cfg)
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        # instantiate model
        self.model = get_model(model_cfg, data_cfg, tokenizer)
        data_module = get_data_module(tokenizer, DictToObject(**data_cfg))
        self.trainset = data_module["train_dataset"]
        self.data_module = data_module
        self.save_path = save_path
        

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.train_cfg.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = self.save_path

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.cuda.current_device()
        self.model.to(device)
        trainer = LLaVATrainer(model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_arguments,
                **self.data_module)
        
        if list(pathlib.Path(self.train_cfg.training_arguments.output_dir).glob("checkpoint-*")):
            results = trainer.train(resume_from_checkpoint=True)
        else:
            results = trainer.train()

        if self.model_cfg.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.training_arguments.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
            if self.training_arguments.local_rank == 0 or self.training_arguments.local_rank == -1:
                self.model.config.save_pretrained(self.training_arguments.output_dir)
                self.model.save_pretrained(self.training_arguments.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(self.training_arguments.output_dir, 'non_lora_trainables.bin'))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=self.training_arguments.output_dir)


        return (
            self.get_parameters({}),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def gen_client_fn(
    tokenizer,
    data_cfg: DictConfig,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    save_path: str,
    partition_id: int = 0,
    api: bool = False,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        json_path = data_cfg.editable_path + str(partition_id+1) + data_cfg.extension
        per_client_data_path = os.path.join(data_cfg.full_path, json_path)
        data_cfg["per_client_data_path"] = per_client_data_path

        return FlowerClient(
            model_cfg,
            train_cfg,
            data_cfg,
            tokenizer,
            save_path,
        ).to_client()

    return client_fn


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
