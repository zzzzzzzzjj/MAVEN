# SA-MDPO for Qwen-VL

> Official implementation of **"[论文标题]"**  
> Under review | Paper link coming soon

---

## Overview

[2-3句话介绍 MAVEN 框架和 SA-MDPO 方法的核心思想]

---

## Requirements

- Python 3.x
- CUDA xx.x

```bash
pip install -r requirements.txt
```

---

## Dataset: MacroValue-Bench

MacroValue-Bench is a macro-societal value evaluation benchmark for vision-language models,
covering six value dimensions: Peace, Development, Equity, Justice, Democracy, and Freedom.

> **Note:** The dataset will be released upon paper acceptance. 
> Please check back later or watch this repository for updates.

---

## Training

### Full Fine-tuning

```bash
bash scripts/finetune.sh
```

### SA-MDPO Fine-tuning

```bash
bash scripts/finetune_sa_mdpo.sh
```

Key arguments:
- `--span_alpha`: ...
- `--dpo_beta`: ...

---

## Evaluation

```bash
bash scripts/evaluate.sh \
    --model_path /path/to/checkpoint \
    --data_path data/test.jsonl
```

Metrics: QWK, Accuracy, F1-macro, VSMS, Recall, Precision  
across six value dimensions: Peace, Development, Equity, Justice, Democracy, Freedom.

---

## Citation

Our paper is currently under review. Citation information will be provided upon publication.

---

## Acknowledgement

This project builds upon
[Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
by [2U1](https://github.com/2U1), licensed under the
[Apache-2.0 License](https://github.com/2U1/Qwen-VL-Series-Finetune/blob/master/LICENSE).  
We reuse and modify portions of the training pipeline and data loading code.  
We thank the author for the excellent open-source implementation.
