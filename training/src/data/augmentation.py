import albumentations as alb
from albumentations.pytorch import ToTensorV2

from .preprocessing import MedicalImagePreprocessor
from ..utils import load_augmentation_config

def get_alzheimer_grayscale_augmentation(
        architecture_name: str,
        dataset_size: int,
        is_training: bool = True
) -> alb.Compose:
    preprocessor = MedicalImagePreprocessor(architecture_name)
    config = preprocessor.config
    aug_config = load_augmentation_config()

    is_transformer = architecture_name.lower() in ['vit_b_16', 'swin_v2_tiny']

    if is_training:
        thresholds = aug_config['dataset_size_thresholds']

        if dataset_size < thresholds['small']:
            print(f"\nAugmentação Pesada: Dataset pequeno (<{thresholds['small']}) - {'Transformer' if is_transformer else 'CNN'}\n")
            if is_transformer:
                cfg = aug_config['train_heavy_transformer']
                augmentations = _get_transformer_heavy_pipeline(config, cfg)
            else:
                cfg = aug_config['train_heavy_cnn']
                augmentations = _get_cnn_heavy_pipeline(config, cfg)

        elif dataset_size < thresholds['medium']:
            print(f"\nAugmentação Moderada: Dataset médio (<{thresholds['medium']}) - {'Transformer' if is_transformer else 'CNN'}\n")
            if is_transformer:
                cfg = aug_config['train_moderate_transformer']
                augmentations = _get_transformer_moderate_pipeline(config, cfg)
            else:
                cfg = aug_config['train_moderate_cnn']
                augmentations = _get_cnn_moderate_pipeline(config, cfg)

        else:
            print(f"\nAugmentação Leve: Dataset grande (≥{thresholds['large']}) - {'Transformer' if is_transformer else 'CNN'}\n")
            if is_transformer:
                cfg = aug_config['train_light_transformer']
                augmentations = _get_transformer_light_pipeline(config, cfg)
            else:
                cfg = aug_config['train_light_cnn']
                augmentations = _get_cnn_light_pipeline(config, cfg)
    else:
        augmentations = [
            alb.Resize(config["image_size"], config["image_size"]),
            alb.Normalize(mean=config["mean"], std=config["std"]),
            ToTensorV2()
        ]

    return alb.Compose(augmentations)

def create_synthetic_augmentation_for_minority(architecture_name: str) -> alb.Compose:
    preprocessor = MedicalImagePreprocessor(architecture_name)
    config = preprocessor.config
    aug_config = load_augmentation_config()
    is_transformer = architecture_name.lower() in ['vit_b_16', 'swin_v2_tiny']

    if is_transformer:
        cfg = aug_config['synthetic_transformer']
        augmentations = _get_transformer_synthetic_pipeline(config, cfg)
    else:
        cfg = aug_config['synthetic_cnn']
        augmentations = _get_cnn_synthetic_pipeline(config, cfg)

    return alb.Compose(augmentations)

def _get_transformer_heavy_pipeline(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
        alb.Affine(
            scale=(float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1])),
            translate_percent=(float(cfg['affine']['translate_percent'][0]), float(cfg['affine']['translate_percent'][1])),
            p=cfg['affine']['probability']
        ),
        alb.CLAHE(
            clip_limit=cfg['clahe']['clip_limit'],
            tile_grid_size=(int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1])),
            p=cfg['clahe']['probability']
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast']['brightness_limit'],
            contrast_limit=cfg['brightness_contrast']['contrast_limit'],
            p=cfg['brightness_contrast']['probability']
        ),
        alb.RandomGamma(
            gamma_limit=(float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1])),
            p=cfg['random_gamma']['probability']
        ),
        alb.GaussNoise(
            std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
            mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])),
            p=cfg['gauss_noise']['probability']
        ),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_cnn_heavy_pipeline(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
        alb.Affine(
            scale=(float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1])),
            translate_percent=(float(cfg['affine']['translate_percent'][0]), float(cfg['affine']['translate_percent'][1])),
            p=cfg['affine']['probability']
        ),
        alb.CLAHE(
            clip_limit=cfg['clahe']['clip_limit'],
            tile_grid_size=(int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1])),
            p=cfg['clahe']['probability']
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast']['brightness_limit'],
            contrast_limit=cfg['brightness_contrast']['contrast_limit'],
            p=cfg['brightness_contrast']['probability']
        ),
        alb.RandomGamma(
            gamma_limit=(float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1])),
            p=cfg['random_gamma']['probability']
        ),
        alb.OneOf([
            alb.GaussianBlur(blur_limit=(int(cfg['blur_sharpen']['gaussian_blur']['blur_limit'][0]), int(cfg['blur_sharpen']['gaussian_blur']['blur_limit'][1])), p=1.0),
            alb.Sharpen(
                alpha=(float(cfg['blur_sharpen']['sharpen']['alpha'][0]), float(cfg['blur_sharpen']['sharpen']['alpha'][1])),
                lightness=(float(cfg['blur_sharpen']['sharpen']['lightness'][0]), float(cfg['blur_sharpen']['sharpen']['lightness'][1])),
                p=1.0
            ),
        ], p=cfg['blur_sharpen']['probability']),
        alb.GaussNoise(
            std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
            mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])),
            p=cfg['gauss_noise']['probability']
        ),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_transformer_moderate_pipeline(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
        alb.Affine(
            scale=(float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1])),
            translate_percent=(float(cfg['affine']['translate_percent'][0]), float(cfg['affine']['translate_percent'][1])),
            p=cfg['affine']['probability']
        ),
        alb.CLAHE(
            clip_limit=cfg['clahe']['clip_limit'],
            tile_grid_size=(int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1])),
            p=cfg['clahe']['probability']
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast']['brightness_limit'],
            contrast_limit=cfg['brightness_contrast']['contrast_limit'],
            p=cfg['brightness_contrast']['probability']
        ),
        alb.RandomGamma(
            gamma_limit=(float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1])),
            p=cfg['random_gamma']['probability']
        ),
        alb.GaussNoise(
            std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
            mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])),
            p=cfg['gauss_noise']['probability']
        ),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_cnn_moderate_pipeline(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
        alb.CLAHE(
            clip_limit=cfg['clahe']['clip_limit'],
            tile_grid_size=(int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1])),
            p=cfg['clahe']['probability']
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast']['brightness_limit'],
            contrast_limit=cfg['brightness_contrast']['contrast_limit'],
            p=cfg['brightness_contrast']['probability']
        ),
        alb.RandomGamma(
            gamma_limit=(float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1])),
            p=cfg['random_gamma']['probability']
        ),
        alb.GaussianBlur(blur_limit=(int(cfg['gaussian_blur']['blur_limit'][0]), int(cfg['gaussian_blur']['blur_limit'][1])), p=cfg['gaussian_blur']['probability']),
        alb.GaussNoise(
            std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
            mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])),
            p=cfg['gauss_noise']['probability']
        ),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_transformer_light_pipeline(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
        alb.CLAHE(
            clip_limit=cfg['clahe']['clip_limit'],
            tile_grid_size=(int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1])),
            p=cfg['clahe']['probability']
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast']['brightness_limit'],
            contrast_limit=cfg['brightness_contrast']['contrast_limit'],
            p=cfg['brightness_contrast']['probability']
        ),
        alb.RandomGamma(
            gamma_limit=(float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1])),
            p=cfg['random_gamma']['probability']
        ),
        alb.GaussNoise(
            std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
            mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])),
            p=cfg['gauss_noise']['probability']
        ),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_cnn_light_pipeline(config, cfg):
    return [
        alb.Resize(config["image_size"], config["image_size"]),
        alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
        alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
        alb.CLAHE(
            clip_limit=cfg['clahe']['clip_limit'],
            tile_grid_size=(int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1])),
            p=cfg['clahe']['probability']
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast']['brightness_limit'],
            contrast_limit=cfg['brightness_contrast']['contrast_limit'],
            p=cfg['brightness_contrast']['probability']
        ),
        alb.GaussNoise(
            std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
            mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])),
            p=cfg['gauss_noise']['probability']
        ),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_transformer_synthetic_pipeline(config, cfg):
    return [
        alb.Affine(
            translate_percent={"x": (float(cfg['affine']['translate_percent']['x'][0]), float(cfg['affine']['translate_percent']['x'][1])),
                              "y": (float(cfg['affine']['translate_percent']['y'][0]), float(cfg['affine']['translate_percent']['y'][1]))},
            scale=(float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1])),
            rotate=(float(cfg['affine']['rotate'][0]), float(cfg['affine']['rotate'][1])),
            p=cfg['affine']['probability']
        ),
        alb.ElasticTransform(alpha=cfg['elastic_transform']['alpha'], sigma=cfg['elastic_transform']['sigma'], p=cfg['elastic_transform']['probability']),
        alb.Perspective(scale=(float(cfg['perspective']['scale'][0]), float(cfg['perspective']['scale'][1])), p=cfg['perspective']['probability']),
        alb.GridDistortion(num_steps=int(cfg['grid_distortion']['num_steps']), distort_limit=float(cfg['grid_distortion']['distort_limit']), p=float(cfg['grid_distortion']['probability'])),
        alb.RandomToneCurve(scale=cfg['random_tone_curve']['scale'], p=cfg['random_tone_curve']['probability']),
        alb.GaussNoise(std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
                      mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])), p=cfg['gauss_noise']['probability']),
        alb.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=0.2),
        alb.Resize(config["image_size"], config["image_size"]),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]

def _get_cnn_synthetic_pipeline(config, cfg):
    return [
        alb.Affine(
            translate_percent={"x": (float(cfg['affine']['translate_percent']['x'][0]), float(cfg['affine']['translate_percent']['x'][1])),
                              "y": (float(cfg['affine']['translate_percent']['y'][0]), float(cfg['affine']['translate_percent']['y'][1]))},
            scale=(float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1])),
            rotate=(float(cfg['affine']['rotate'][0]), float(cfg['affine']['rotate'][1])),
            p=cfg['affine']['probability']
        ),
        alb.ElasticTransform(alpha=cfg['elastic_transform']['alpha'], sigma=cfg['elastic_transform']['sigma'], p=cfg['elastic_transform']['probability']),
        alb.Perspective(scale=(float(cfg['perspective']['scale'][0]), float(cfg['perspective']['scale'][1])), p=cfg['perspective']['probability']),
        alb.RandomToneCurve(scale=cfg['random_tone_curve']['scale'], p=cfg['random_tone_curve']['probability']),
        alb.MotionBlur(blur_limit=5, p=0.15),
        alb.GaussNoise(std_range=(float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1])),
                      mean_range=(float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1])), p=cfg['gauss_noise']['probability']),
        alb.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=0.2),
        alb.Resize(config["image_size"], config["image_size"]),
        alb.Normalize(mean=config["mean"], std=config["std"]),
        ToTensorV2()
    ]