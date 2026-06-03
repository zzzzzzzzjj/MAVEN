import os
import numpy as np
import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

import trl.import_utils as trl_import_utils
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def _normalize_trl_optional_flags():
    for name in dir(trl_import_utils):
        if not (name.startswith("_") and name.endswith("_available")):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, value[0])


_normalize_trl_optional_flags()

from trl import DPOTrainer
from trl.trainer.utils import pad_to_length, flush_left, selective_log_softmax
from train.train_utils import get_peft_state_non_lora_maybe_zero_3


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


class QwenMDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super(QwenMDPOTrainer, self).__init__(*args, **kwargs)
        self.num_levels = getattr(self.args, "num_levels", 2)
        self.implicit_beta = getattr(self.args, "implicit_beta", True)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset


    def concatenated_inputs(
        self,
        batch: dict[str, Union[list, torch.LongTensor]],
        padding_value: int,
    ) -> dict[str, torch.LongTensor]:
        N = self.num_levels
        concatenated_batch = {}

        concatenated_batch['prompt_input_ids'] = torch.cat(
            [batch["prompt_input_ids"]] * N, dim=0
        )
        concatenated_batch['prompt_attention_mask'] = torch.cat(
            [batch["prompt_attention_mask"]] * N, dim=0
        )
        if "prompt_mm_token_type_ids" in batch:
            concatenated_batch["prompt_mm_token_type_ids"] = torch.cat(
                [batch["prompt_mm_token_type_ids"]] * N, dim=0
            )


        if 'pixel_values' in batch:
            concatenated_batch['pixel_values'] = torch.cat(
                [batch["pixel_values"]] * N, dim=0
            )
            concatenated_batch['image_grid_thw'] = torch.cat(
                [batch["image_grid_thw"]] * N, dim=0
            )
        if 'pixel_values_videos' in batch:
            concatenated_batch['pixel_values_videos'] = torch.cat(
                [batch["pixel_values_videos"]] * N, dim=0
            )
            concatenated_batch['video_grid_thw'] = torch.cat(
                [batch["video_grid_thw"]] * N, dim=0
            )
        if "second_per_grid_ts" in batch:
            concatenated_batch["second_per_grid_ts"] = batch["second_per_grid_ts"] * N


        max_completion_length = max(
            batch[f"response_{i}_input_ids"].shape[1] for i in range(N)
        )

        completion_ids_list = []
        completion_mask_list = []
        for i in range(N):
            completion_ids_list.append(
                pad_to_length(batch[f"response_{i}_input_ids"], max_completion_length, pad_value=padding_value)
            )
            completion_mask_list.append(
                pad_to_length(batch[f"response_{i}_attention_mask"], max_completion_length, pad_value=0)
            )

        concatenated_batch['completion_input_ids'] = torch.cat(completion_ids_list, dim=0)
        concatenated_batch['completion_attention_mask'] = torch.cat(completion_mask_list, dim=0)

        return concatenated_batch


    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
    ) -> List[torch.FloatTensor]:
        N = self.num_levels
        num_examples = batch['prompt_input_ids'].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        if 'pixel_values' in batch:
            model_kwargs['pixel_values'] = concatenated_batch['pixel_values']
            model_kwargs['image_grid_thw'] = concatenated_batch['image_grid_thw']
        if 'pixel_values_videos' in batch:
            model_kwargs['pixel_values_videos'] = concatenated_batch['pixel_values_videos']
            model_kwargs['video_grid_thw'] = concatenated_batch['video_grid_thw']
        if "second_per_grid_ts" in batch:
            model_kwargs["second_per_grid_ts"] = concatenated_batch["second_per_grid_ts"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        prompt_mm_token_type_ids = concatenated_batch.get("prompt_mm_token_type_ids")

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        if prompt_mm_token_type_ids is not None:
            completion_mm_token_type_ids = torch.zeros_like(completion_input_ids)
            mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, completion_mm_token_type_ids), dim=1)
            attention_mask, input_ids, loss_mask, mm_token_type_ids = flush_left(
                attention_mask, input_ids, loss_mask, mm_token_type_ids
            )
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if "ipo" in self.loss_type:
            all_logps = per_token_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            all_logps = per_token_logps.sum(-1)

        logp_list = [
            all_logps[i * num_examples: (i + 1) * num_examples]
            for i in range(N)
        ]
        return logp_list


    def multi_level_dpo_loss(
        self,
        policy_logps_list: List[torch.FloatTensor],
        reference_logps_list: List[torch.FloatTensor],
        implicit_beta: bool = True,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        losses_list = []
        chosen_rewards_list = []
        rejected_rewards_list = []

        for i in range(len(policy_logps_list) - 1):
            for j in range(i + 1, len(policy_logps_list)):
                pi_logratios = (policy_logps_list[i] - policy_logps_list[j]).to(self.accelerator.device)
                ref_logratios = (reference_logps_list[i] - reference_logps_list[j]).to(self.accelerator.device)
                logits = pi_logratios - ref_logratios

                if implicit_beta:
                    beta_used = self.beta * (1 + 0.5 * (1 - np.exp(-0.5 * (j - i))))
                else:
                    beta_used = self.beta

                loss = -F.logsigmoid(beta_used * logits) * (1 - self.label_smoothing)

                if i == 0:
                    kl_penalty = (
                        reference_logps_list[i].to(self.accelerator.device)
                        - policy_logps_list[i].to(self.accelerator.device)
                    )
                    loss = loss + kl_penalty

                chosen_rewards = (
                    beta_used * (
                        policy_logps_list[i].to(self.accelerator.device)
                        - reference_logps_list[i].to(self.accelerator.device)
                    )
                ).detach()
                rejected_rewards = (
                    beta_used * (
                        policy_logps_list[j].to(self.accelerator.device)
                        - reference_logps_list[j].to(self.accelerator.device)
                    )
                ).detach()

                losses_list.append(loss)
                chosen_rewards_list.append(chosen_rewards)
                rejected_rewards_list.append(rejected_rewards)

        losses = torch.cat(losses_list, dim=0)
        chosen_rewards = torch.cat(chosen_rewards_list, dim=0)
        rejected_rewards = torch.cat(rejected_rewards_list, dim=0)

        accuracies = (chosen_rewards > rejected_rewards).float()

        metrics = {
            "rewards/accuracies": accuracies.mean().cpu(),
            "rewards/chosen": chosen_rewards.mean().cpu(),
            "rewards/rejected": rejected_rewards.mean().cpu(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().cpu(),
        }

        return losses.mean(), metrics


    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        metrics = {}

        policy_list = self.concatenated_forward(model, batch)

        with torch.no_grad():
            if self.ref_model is None:
                with model.disable_adapter():
                    reference_list = self.concatenated_forward(model, batch)
            else:
                reference_list = self.concatenated_forward(self.ref_model, batch)


        loss, batch_metrics = self.multi_level_dpo_loss(
            policy_list, reference_list, self.implicit_beta
        )

        prefix = "eval_" if train_eval == "eval" else ""
        metrics.update({f"{prefix}{k}": v for k, v in batch_metrics.items()})

        return loss, metrics


    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "merger" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in decay_parameters and n not in special_lr_parameters and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n not in decay_parameters and n not in special_lr_parameters and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]
                if visual_parameters:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if n in decay_parameters and n in visual_parameters and p.requires_grad],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_lr,
                            "param_group_name": "visual_decay",
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if n not in decay_parameters and n in visual_parameters and p.requires_grad],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_lr,
                            "param_group_name": "visual_non_decay",
                        },
                    ])
                if merger_parameters:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if n in decay_parameters and n in merger_parameters and p.requires_grad],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.merger_lr,
                            "param_group_name": "merger_decay",
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters()
                                       if n not in decay_parameters and n in merger_parameters and p.requires_grad],
                            "weight_decay": 0.0,
                            "lr": self.args.merger_lr,
                            "param_group_name": "merger_non_decay",
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in decay_parameters and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n not in decay_parameters and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(), require_grad_only=True
        )
        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))
