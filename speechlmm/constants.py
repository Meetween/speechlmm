CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

LORA_ADAPTERS_DIR = "lora_adapters"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200  # deprecated
AUDIO_TOKEN_INDEX = -300  # deprecated
VIDEO_TOKEN_INDEX = -400

DEFAULT_IMAGE_TOKEN = "<image>"  # deprecated
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"  # deprecated

DEFAULT_AUDIO_TOKEN = "<audio>"  # deprecated
DEFAULT_AUDIO_INPUT_START_TOKEN = "<audio_input_start>"
DEFAULT_AUDIO_INPUT_END_TOKEN = "<audio_input_end>"
AUDIO_PLACEHOLDER = "<audio-placeholder>"  # deprecated

DEFAULT_AUDIO_OUTPUT_START_TOKEN = "<audio_output_start>"
DEFAULT_AUDIO_OUTPUT_END_TOKEN = "<audio_output_end>"

DEFAULT_AUDIO_CONDITION_START_TOKEN = "<audio_condition_start>"
DEFAULT_AUDIO_CONDITION_END_TOKEN = "<audio_condition_end>"

DEFAULT_AUDIO_PAD_TOKEN = "<PAD_AUDIO>"
DEFAULT_AUDIO_EPAD_TOKEN = "<EPAD_AUDIO>"

DEFAULT_VIDEO_TOKEN = "<video>"  # deprecated
DEFAULT_VIDEO_INPUT_START_TOKEN = "<video_input_start>"
DEFAULT_VIDEO_INPUT_END_TOKEN = "<video_input_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"  # deprecated
