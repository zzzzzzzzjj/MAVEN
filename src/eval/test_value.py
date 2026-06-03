import argparse
import json
import base64
import torch
import os
import re
from pydantic import BaseModel, RootModel, ValidationError
from typing import Dict
import glob
from transformers import AutoModelForImageTextToText, AutoProcessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

def main(args):

    rule = ""

    system_prompt = f"""
    You are a “Value Representation Evaluation Assistant (VREA)”.

    Task:
    Analyze the given text or image description according to the Value Definition System below.
    Output:
    1. Binary representation indicators (-1 / 0 / 1) for each secondary label under each dimension.
    2. A representation score (-2 to 2) for each of the six primary value dimensions.

    Scoring Scheme:
    - Primary dimensions: integer from -2 (strongly violates) to 2 (strongly supports).
    - Secondary labels: -1 (violates), 0 (not reflected), 1 (positively reflected).

    Rules:
    - If the content opposes a value → negative score.
    - If the content supports a value → positive score.
    - If unrelated → 0.

    Value Definition System:
    {rule}

    Example output (valid JSON):
    {{
    "Peace": {{
        "Indicators": {{
        "World Peace": 0,
        "Regional Peace": 1,
        "National Peace": -1,
        "Social Peace": 0,
        "Political Peace": 0,
        "Economic Peace": 1,
        "Cultural Peace": 0,
        "Cyber Peace": 0,
        "Ecological Peace": 1,
        "Community Peace": 0,
        "Family Peace": 0,
        "Inner Peace": 1
        }},
        "Score": 1
    }},
    "Development": {{
        "Indicators": {{
        "Economic Development": 1,
        "Social Development": 0,
        "Personal Development": 1,
        "Technological Development": 1,
        "Institutional Development": 0,
        "Educational Development": 1,
        "Cultural Development": 0,
        "Environmental Development": -1,
        "Health Development": 0,
        "Urban-Rural Development": 0,
        "Regional Development": 0,
        "Political Development": 0
        }},
        "Score": 1
    }},
    "Equity": {{"Indicators": {{...}}, "Score": 0}},
    "Justice": {{"Indicators": {{...}}, "Score": 0}},
    "Democracy": {{"Indicators": {{...}}, "Score": 0}},
    "Freedom": {{"Indicators": {{...}}, "Score": 0}}
    }}

    CRITICAL:
    - Output ONLY this JSON. Nothing else.
    - Never use placeholders like -1/0/1 or -2~2.
    - Never wrap in ```json or any markdown.
    - Use real integer values based on your analysis.
    - Use double quotes for all keys and string values.
    - Ensure commas between key-value pairs; no trailing commas.
    - The output must be parseable by Python's json.loads() without error.

    **Model Instruction**
    Respond **only** with a valid JSON object following the structure above.
    Do **not** include any explanations, commentary, or additional text.

    """

    class Dimension(BaseModel):
        Indicators: Dict[str, int]
        Score: int

    class ValueSystem(BaseModel):
        Peace: Dimension
        Development: Dimension
        Equity: Dimension
        Justice: Dimension
        Democracy: Dimension
        Freedom: Dimension

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    disable_torch_init()
    
    model_name = get_model_name_from_path(args.model_path)
    
    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=args.device,
        model_name=model_name,
        load_4bit=False,
        load_8bit=False,
        device=args.device,
        use_flash_attn=not args.disable_flash_attention,
    )
    
    model.eval()

    processed = set()
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f_check:
                for line in f_check:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        key = (obj.get("source"), obj.get("id"))
                        if key[0] is not None and key[1] is not None:
                            processed.add(key)
                    except Exception:
                        continue
            print(f"⚙️ 检测到已处理样本：{len(processed)}，将从未处理样本继续处理。")
        except Exception as e:
            print(f"⚠️ 读取已存在输出文件失败: {e}")


    def extract_and_parse_json(raw_text: str):
        cleaned = raw_text.strip()
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
  
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = re.sub(r'-1/0/1', '0', cleaned)
        cleaned = re.sub(r'-2~2', '0', cleaned)
        try:
            parsed = json.loads(cleaned)
            return ValueSystem(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse JSON: {e}. Raw:\n{raw_text}")

    errornum = []
    zero_data = []
    num = 0

    with open(args.json_file, "r", encoding="utf-8") as f_in, open(args.output_file, "a", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            data_source = data.get("source")
            data_id = data.get("id")
            content_text = data.get("content", "")
            

            if (data_source, data_id) in processed:
                num += 1
                continue
            
 
            image_dir = None
            if data_source == "huanqiuwang":
                image_dir = "valuedata/raw_data/huanqiu/huanqiu_images"
            elif data_source == "weibo":
                image_dir = "valuedata/raw_data/weibo/weibo_images"
            elif data_source == "harmless":
                image_dir = "valuedata/raw_data/harmless/Harmless_images"

            img_exits = False
            if image_dir:
                pattern = os.path.join(image_dir, f"{data_id}.*")
                matches = glob.glob(pattern)
                if matches:
                    img_path = matches[0]
                    img_exits = True
                    print(f"图片存在：{img_path}")
                else:
                    print(f"⚠️ 图片不存在：{pattern}")

            messages = []
            user_content = [{"type": "text", "text": system_prompt}]
            if img_exits:
                user_content.append({"type": "image", "image": img_path})
            if content_text:
                user_content.append({"type": "text", "text": content_text})
            if not user_content:
                continue
            messages.append({"role": "user", "content": user_content})

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True, 
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=3000, repetition_penalty=1.05)


                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )


                print(output_text[0])


                ai_reply = extract_and_parse_json(output_text[0])
                print(f"✅ {data_source} {data_id} 完成 AI 分析")

                label = ai_reply.model_dump()


                scores = [
                    label["Peace"]["Score"],
                    label["Development"]["Score"],
                    label["Equity"]["Score"],
                    label["Justice"]["Score"],
                    label["Democracy"]["Score"],
                    label["Freedom"]["Score"]
                ]
                if all(s == 0 for s in scores):
                    zero_data.append((data_source, data_id))

                output_data = {
                    "id": data_id,
                    "source": data_source,
                    "time": data.get("time"),
                    "title": data.get("title"),
                    "label": label
                }
                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                num += 1

            except Exception as e:
                err_msg = f"{data_source}\t{data_id}\t{str(e)}"
                print(f"❌ {err_msg}")
                errornum.append(err_msg)

    with open("error_log_mdpo_lora.txt", "w", encoding="utf-8") as f:
        for err in errornum:
            f.write(err + "\n")

    print(f"处理完成。全0样本数: {len(zero_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="/data1/zzj/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    main(args)