import os
from typing import Dict, List, Optional
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from params import DataArguments
from constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import (
    chat_template_uses_reasoning_prefill,
    format_assistant_response,
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    get_video_info,
    model_supports_optional_reasoning,
    pad_sequence,
    replace_image_tokens,
    use_default_system_message,
)


class MDPODataset(Dataset):

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        num_levels: int = 2,
        padding: bool = True,
    ):
        super(MDPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.num_levels = num_levels

        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        self.model_type, self.image_patch_size, self.return_video_metadata = get_qwen_multimodal_settings(
            self.model_id
        )
        self.reasoning_supported = chat_template_uses_reasoning_prefill(self.processor, self.model_type)
        self.use_reasoning_prefill = self.data_args.enable_reasoning and self.reasoning_supported
        self.optional_reasoning_supported = model_supports_optional_reasoning(self.model_type)
        if self.data_args.enable_reasoning and not self.reasoning_supported:
            raise ValueError(
                f"`enable_reasoning` is only supported for Qwen3-VL Thinking or Qwen3.5 models. "
                f"Current model_type={self.model_type!r} does not qualify."
            )

    def __len__(self):
        return len(self.list_data_dict)

    def _get_responses(self, sources: dict) -> tuple[List[str], List[Optional[str]]]:
        if "responses" in sources:
            responses = sources["responses"]
            assert len(responses) == self.num_levels, (
                f"样本中 responses 列表长度 {len(responses)} 与 num_levels={self.num_levels} 不符。"
            )
            reasonings = sources.get("reasonings", [None] * self.num_levels)
            if len(reasonings) != self.num_levels:
                raise ValueError("reasonings 列表长度必须与 responses 一致。")
        elif "chosen" in sources and "rejected" in sources:
            assert self.num_levels == 2, (
                "检测到旧格式（chosen/rejected），但 num_levels != 2，请检查数据格式。"
            )
            responses = [sources["chosen"], sources["rejected"]]
            reasonings = [
                sources.get("chosen_reasoning"),
                sources.get("rejected_reasoning"),
            ]
        else:
            raise KeyError("样本中必须包含 'responses' 字段（或兼容旧格式的 'chosen'/'rejected'）。")

        return responses, reasonings

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False
        processor = self.processor

        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                
                    if not os.path.exists(image_file):
                        print(f"[图像不存在] id={sources.get('id','?')} path={image_file}")
                        pixel_key = None
                        grid_key = None
                        break

                images.append(get_image_info(
                    image_file,
                    self.image_min_pixel, self.image_max_pixel,
                    self.image_resized_w, self.image_resized_h,
                    self.image_patch_size,
                ))

        elif "video" in sources:
            is_video = True
            images = None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder
            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file,
                    self.video_min_pixel, self.video_max_pixel,
                    self.video_resized_w, self.video_resized_h,
                    self.data_args.fps, self.image_patch_size,
                    return_video_metadata=self.return_video_metadata,
                )
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images = None
            videos = None


        responses, reasonings = self._get_responses(sources)

        has_reasonings = [isinstance(r, str) and r.strip() for r in reasonings]
        if self.data_args.enable_reasoning:
            if len(set(has_reasonings)) > 1:
                raise ValueError(
                    "多级 DPO 样本中，所有 responses 必须同时提供 reasoning，或都不提供。"
                )
            if (
                self.reasoning_supported
                and not self.optional_reasoning_supported
                and not has_reasonings[0]
            ):
                raise ValueError(
                    "Qwen3-VL Thinking 的 DPO 样本必须包含 reasonings。"
                )

        use_reasoning_prefill = self.use_reasoning_prefill and has_reasonings[0]
        use_closed_think_prefill = self.optional_reasoning_supported and not use_reasoning_prefill

        assistant_prefill, cleaned_responses = format_assistant_response(
            responses[0],
            reasonings[0],
            enable_reasoning=self.data_args.enable_reasoning,
            use_reasoning_prefill=use_reasoning_prefill,
            use_closed_think_prefill=use_closed_think_prefill,
        )
        cleaned_responses = [cleaned_responses]

        for resp, reas in zip(responses[1:], reasonings[1:]):
            _, cleaned = format_assistant_response(
                resp,
                reas,
                enable_reasoning=self.data_args.enable_reasoning,
                use_reasoning_prefill=use_reasoning_prefill,
                use_closed_think_prefill=use_closed_think_prefill,
            )
            cleaned_responses.append(cleaned)

        all_input_ids = []
        all_prompt_mm_token_type_ids = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_grid = []

        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_ids = processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors='pt'
            )['input_ids']
            all_input_ids.append(system_ids.squeeze(0))
            all_prompt_mm_token_type_ids.append(
                torch.zeros_like(system_ids, dtype=torch.long).squeeze(0)
            )

        user_prompt = replace_image_tokens(sources["prompt"], is_video=is_video)
        user_input = (
            f"{DEFAULT_IM_START_TOKEN}user\n{user_prompt}{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}assistant\n{assistant_prefill}"
        )

        if DEFAULT_IMAGE_TOKEN in user_input:
            inputs = processor(
                text=[user_input], images=images, videos=videos,
                padding=False, do_resize=False, return_tensors='pt'
            )
            prompt_input_ids = inputs['input_ids']
            prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])

        elif DEFAULT_VIDEO_TOKEN in user_input:
            if self.model_type == "qwen2_5_vl":
                inputs = processor(
                    text=[user_input], images=images, videos=videos,
                    padding=False, do_resize=False, return_tensors='pt', **video_kwargs
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                all_second_grid.extend(inputs["second_per_grid_ts"])

            elif self.model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
                video_datas, video_metadatas = zip(*videos)
                inputs = processor(
                    text=[user_input], images=images, videos=list(video_datas),
                    padding=False, do_resize=False, return_tensors='pt',
                    **video_kwargs, video_metadata=list(video_metadatas),
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
            else:
                inputs = processor(
                    text=[user_input], images=images, videos=videos,
                    padding=False, do_resize=False, return_tensors='pt'
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])

            prompt_input_ids = inputs['input_ids']
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])

        else:
            prompt_input_ids = processor.tokenizer(
                user_input, add_special_tokens=False, padding=False, return_tensors='pt'
            )['input_ids']
            prompt_mm_token_type_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

        all_input_ids.append(prompt_input_ids.squeeze(0))
        all_prompt_mm_token_type_ids.append(prompt_mm_token_type_ids.squeeze(0))

        response_input_ids_list = []
        for resp in cleaned_responses:
            resp_text = f"{resp}{DEFAULT_IM_END_TOKEN}\n"
            resp_ids = processor.tokenizer(
                resp_text, add_special_tokens=False, padding=False, return_tensors='pt'
            )['input_ids'].squeeze(0)
            response_input_ids_list.append(resp_ids)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        prompt_mm_token_type_ids = torch.cat(all_prompt_mm_token_type_ids, dim=0).to(torch.long)

        data_dict = dict(
            prompt_input_ids=input_ids,
            prompt_mm_token_type_ids=prompt_mm_token_type_ids,
        )

        for i, resp_ids in enumerate(response_input_ids_list):
            data_dict[f"response_{i}_input_ids"] = resp_ids

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            data_dict[pixel_key] = torch.cat(all_pixel_values, dim=0)
            data_dict[grid_key] = torch.cat(all_image_grid_thw, dim=0)

        if len(all_second_grid) > 0:
            data_dict["second_per_grid_ts"] = all_second_grid

        return data_dict


class DataCollatorForMDPODataset(object):

    def __init__(self, pad_token_id: int, num_levels: int = 2):
        self.pad_token_id = pad_token_id
        self.num_levels = num_levels

    def __call__(self, examples):
        batch_input_ids = []
        batch_prompt_mm_token_type_ids = []
        batch_response_ids = [[] for _ in range(self.num_levels)]

        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_image_thw = []
        batch_video_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()

            batch_input_ids.append(example["prompt_input_ids"])
            batch_prompt_mm_token_type_ids.append(example["prompt_mm_token_type_ids"])

            for i in range(self.num_levels):
                batch_response_ids[i].append(example[f"response_{i}_input_ids"])

            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        prompt_input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        prompt_mm_token_type_ids = pad_sequence(
            batch_prompt_mm_token_type_ids, padding_side='right', padding_value=0
        )
        prompt_attention_mask = (prompt_input_ids != self.pad_token_id).long()

        data_dict = {
            'prompt_input_ids': prompt_input_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'prompt_mm_token_type_ids': prompt_mm_token_type_ids,
        }

        for i in range(self.num_levels):
            padded = pad_sequence(
                batch_response_ids[i], padding_side='right', padding_value=self.pad_token_id
            )
            data_dict[f"response_{i}_input_ids"] = padded
            data_dict[f"response_{i}_attention_mask"] = (padded != self.pad_token_id).long()


        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_video_values) > 0:
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_video_values, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict


def make_mdpo_data_module(model_id, processor, data_args, num_levels: int = 2):
    dpo_dataset = MDPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
        num_levels=num_levels,
    )
    data_collator = DataCollatorForMDPODataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        num_levels=num_levels,
    )
    return dict(
        train_dataset=dpo_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
