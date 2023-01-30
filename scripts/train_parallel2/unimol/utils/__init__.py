from .config import get_config, config
from .seed import seed_all
from .device_adapter import get_device_num, get_device_id, get_job_id, get_rank_id
from .checkpoints import set_save_ckpt_dir