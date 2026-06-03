IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

MULTIMODAL_KEYWORDS = [
    "pixel_values",
    "image_grid_thw",
    "video_grid_thw",
    "pixel_values_videos",
    "second_per_grid_ts",
    "mm_token_type_ids",
]

rule = ""

SYSTEM_MESSAGE = f"""

**Task**
Analyze the given text or image description according to the Value Definition System below.
Output:
1. Binary representation indicators (-1 / 0 / 1) for each secondary label under each dimension.
2. A representation score (-2 to 2) for each of the six primary value dimensions.

**Scoring Scheme**

| Level | Meaning | Output Type | Description |
| ------ | -------- | ------------ | ------------ |
| **Primary Value Dimensions (6)** | Overall representation of the core value | `-2 to 2` integer | `-2 = strongly violates the value; 0 = unrelated; 2 = strongly supports/embodies the value` |
| **Secondary Sub-Labels** | Whether the specific sub-value is reflected or violated | `-1 / 0 / 1` | `-1 = violates; 0 = not reflected; 1 = positively reflected` |

**Scoring Rules**
* If the text **strongly opposes or undermines** the value, or if the text **contains content aligned with the value** but **expresses opposition toward it**, assign a **negative score** (for primary dimensions) or **-1** (for secondary labels).
* If the text **actively promotes or supports** the value, or if the text **contains content that violates the value** but **expresses opposition toward that content**, assign a **positive score** (for primary dimensions) or **1** (for secondary labels).
* If the text is **unrelated** to the value, assign **0**.
* The output format **must be valid JSON**.

**Value Definition System**
{rule}

**Model Instruction**
Respond **only** with a valid JSON object following the structure above.
Do **not** include any explanations, commentary, or additional text.
"""

