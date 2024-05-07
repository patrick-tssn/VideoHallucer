import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    if image_tower.endswith('LanguageBind_Image'):
        local_file = "checkpoints/VideoLLaVA/LanguageBind_Image"
        if not os.path.exists(local_file):
            return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
        else:
            return LanguageBindImageTower(local_file, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        local_file = "checkpoints/VideoLLaVA/LanguageBind_Video_merge"
        if not os.path.exists(local_file):
            return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
        else:
            return LanguageBindVideoTower(local_file, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
