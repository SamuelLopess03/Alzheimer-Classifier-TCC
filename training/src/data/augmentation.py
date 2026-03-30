import albumentations as alb
from albumentations.pytorch import ToTensorV2

from .preprocessing import MedicalImagePreprocessor
from ..utils import load_augmentation_config

_TRANSFORMER_ARCHS = {'vit_b_16', 'swin_v2_tiny'}

def get_alzheimer_grayscale_augmentation(
        architecture_name: str,
        dataset_size: int,
        is_training: bool = True
) -> alb.Compose:
    preprocessor = MedicalImagePreprocessor(architecture_name)
    config = preprocessor.config
    aug_config = load_augmentation_config()
    is_transformer = architecture_name.lower() in _TRANSFORMER_ARCHS

    if not is_training:
        return alb.Compose([
            alb.Resize(config["image_size"], config["image_size"]),
            alb.Normalize(mean=config["mean"], std=config["std"]),
            ToTensorV2()
        ])

    thresholds = aug_config['dataset_size_thresholds']
    family = "transformer" if is_transformer else "cnn"

    if dataset_size < thresholds['small']:
        level = "heavy"
    elif dataset_size < thresholds['medium']:
        level = "moderate"
    else:
        level = "light"

    cfg_key = f"train_{level}_{family}"
    cfg = aug_config[cfg_key]

    label = "Transformer" if is_transformer else "CNN"
    print(f"\nAugmentação {level.capitalize()} ({label}, N={dataset_size})\n")

    pipeline_fn = _PIPELINE_REGISTRY[(family, level)]
    return alb.Compose(pipeline_fn(config, cfg))

def create_synthetic_augmentation_for_minority(architecture_name: str) -> alb.Compose:
    preprocessor = MedicalImagePreprocessor(architecture_name)
    config = preprocessor.config
    aug_config = load_augmentation_config()
    is_transformer = architecture_name.lower() in _TRANSFORMER_ARCHS

    family = "transformer" if is_transformer else "cnn"
    cfg = aug_config[f'synthetic_{family}']
    pipeline_fn = _PIPELINE_REGISTRY[(family, "synthetic")]
    return alb.Compose(pipeline_fn(config, cfg))

def _resize_normalize(config: dict) -> list:
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _flip_rotate(cfg: dict) -> list:
    return [
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
    ]

def _affine(cfg: dict) -> alb.Affine:
    c = cfg['affine']
    return alb.Affine(
        scale=(float(c['scale'][0]), float(c['scale'][1])),
        translate_percent=(float(c['translate_percent'][0]), float(c['translate_percent'][1])),
        p=c['probability']
    )

def _clahe(cfg: dict) -> alb.CLAHE:
    c = cfg['clahe']
    return alb.CLAHE(
        clip_limit=c['clip_limit'],
        tile_grid_size=(int(c['tile_grid_size'][0]), int(c['tile_grid_size'][1])),
        p=c['probability']
    )

def _brightness_contrast(cfg: dict) -> alb.RandomBrightnessContrast:
    c = cfg['brightness_contrast']
    return alb.RandomBrightnessContrast(
        brightness_limit=c['brightness_limit'],
        contrast_limit=c['contrast_limit'],
        p=c['probability']
    )

def _gamma(cfg: dict) -> alb.RandomGamma:
    c = cfg['random_gamma']
    return alb.RandomGamma(
        gamma_limit=(float(c['gamma_limit'][0]), float(c['gamma_limit'][1])),
        p=c['probability']
    )

def _gauss_noise(cfg: dict) -> alb.GaussNoise:
    c = cfg['gauss_noise']
    return alb.GaussNoise(
        std_range=(float(c['std_range'][0]), float(c['std_range'][1])),
        mean_range=(float(c['mean_range'][0]), float(c['mean_range'][1])),
        p=c['probability']
    )

def _motion_blur(cfg: dict) -> alb.MotionBlur:
    c = cfg['motion_blur']
    return alb.MotionBlur(
        blur_limit=int(c['blur_limit']),
        p=c['probability']
    )
    
def _mult_noise(cfg: dict) -> alb.MultiplicativeNoise:
    c = cfg['multiplicative_noise']
    return alb.MultiplicativeNoise(
        multiplier=(float(c['multiplier'][0]), float(c['multiplier'][1])),
        per_channel=bool(c['per_channel']),
        p=c['probability']
    )

def _gaussian_blur(cfg: dict) -> alb.GaussianBlur:
    c = cfg['gaussian_blur']
    return alb.GaussianBlur(
        blur_limit=(int(c['blur_limit'][0]), int(c['blur_limit'][1])),
        p=c['probability']
    )

def _elastic_transform(cfg: dict) -> alb.ElasticTransform:
    c = cfg['elastic_transform']
    return alb.ElasticTransform(
        alpha=c['alpha'],
        sigma=c['sigma'],
        p=c['probability']
    )

def _perspective(cfg: dict) -> alb.Perspective:
    c = cfg['perspective']
    return alb.Perspective(
        scale=(float(c['scale'][0]), float(c['scale'][1])),
        p=c['probability']
    )

def _grid_distortion(cfg: dict) -> alb.GridDistortion:
    c = cfg['grid_distortion']
    return alb.GridDistortion(
        num_steps=int(c['num_steps']),
        distort_limit=float(c['distort_limit']),
        p=float(c['probability'])
    )

def _random_tone_curve(cfg: dict) -> alb.RandomToneCurve:
    c = cfg['random_tone_curve']
    return alb.RandomToneCurve(
        scale=c['scale'],
        p=c['probability']
    )

def _blur_sharpen_oneof(cfg: dict) -> alb.OneOf:
    bs = cfg['blur_sharpen']
    return alb.OneOf([
        alb.GaussianBlur(
            blur_limit=(int(bs['gaussian_blur']['blur_limit'][0]), int(bs['gaussian_blur']['blur_limit'][1])),
            p=1.0
        ),
        alb.Sharpen(
            alpha=(float(bs['sharpen']['alpha'][0]), float(bs['sharpen']['alpha'][1])),
            lightness=(float(bs['sharpen']['lightness'][0]), float(bs['sharpen']['lightness'][1])),
            p=1.0
        ),
    ], p=bs['probability'])

def _affine_synthetic(cfg: dict) -> alb.Affine:
    c = cfg['affine']
    return alb.Affine(
        translate_percent={
            "x": (float(c['translate_percent']['x'][0]), float(c['translate_percent']['x'][1])),
            "y": (float(c['translate_percent']['y'][0]), float(c['translate_percent']['y'][1]))
        },
        scale=(float(c['scale'][0]), float(c['scale'][1])),
        rotate=(float(c['rotate'][0]), float(c['rotate'][1])),
        p=c['probability']
    )

def _cnn_heavy(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        *_flip_rotate(cfg),
        _affine(cfg),
        _clahe(cfg),
        _brightness_contrast(cfg),
        _gamma(cfg),
        _blur_sharpen_oneof(cfg),
        _gauss_noise(cfg),
        *_resize_normalize(config)[1:]   # Normalize + ToTensorV2 (Resize já feito)
    ]

def _cnn_moderate(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        *_flip_rotate(cfg),
        _clahe(cfg),
        _brightness_contrast(cfg),
        _gamma(cfg),
        _gaussian_blur(cfg),
        _gauss_noise(cfg),
        *_resize_normalize(config)[1:]
    ]

def _cnn_light(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        *_flip_rotate(cfg),
        _clahe(cfg),
        _brightness_contrast(cfg),
        _gauss_noise(cfg),
        *_resize_normalize(config)[1:]
    ]

def _transformer_heavy(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        *_flip_rotate(cfg),
        _affine(cfg),
        _clahe(cfg),
        _brightness_contrast(cfg),
        _gamma(cfg),
        _gauss_noise(cfg),
        *_resize_normalize(config)[1:]
    ]

def _transformer_moderate(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        *_flip_rotate(cfg),
        _affine(cfg),
        _clahe(cfg),
        _brightness_contrast(cfg),
        _gamma(cfg),
        _gauss_noise(cfg),
        *_resize_normalize(config)[1:]
    ]

def _transformer_light(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        *_flip_rotate(cfg),
        _clahe(cfg),
        _brightness_contrast(cfg),
        _gamma(cfg),
        _gauss_noise(cfg),
        *_resize_normalize(config)[1:]
    ]

def _cnn_synthetic(config, cfg):
    return [
        _affine_synthetic(cfg),
        _elastic_transform(cfg),
        _perspective(cfg),
        _random_tone_curve(cfg),
        _motion_blur(cfg),
        _gauss_noise(cfg),
        _mult_noise(cfg),
        *_resize_normalize(config)
    ]

def _transformer_synthetic(config, cfg):
    return [
        _affine_synthetic(cfg),
        _elastic_transform(cfg),
        _perspective(cfg),
        _grid_distortion(cfg),
        _random_tone_curve(cfg),
        _gauss_noise(cfg),
        _mult_noise(cfg),
        *_resize_normalize(config)
    ]

_PIPELINE_REGISTRY = {
    ("cnn", "heavy"):      _cnn_heavy,
    ("cnn", "moderate"):   _cnn_moderate,
    ("cnn", "light"):      _cnn_light,
    ("cnn", "synthetic"):  _cnn_synthetic,
    ("transformer", "heavy"):     _transformer_heavy,
    ("transformer", "moderate"):  _transformer_moderate,
    ("transformer", "light"):     _transformer_light,
    ("transformer", "synthetic"): _transformer_synthetic,
}