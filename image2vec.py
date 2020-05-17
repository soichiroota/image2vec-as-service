from tensorflow.keras.applications import ( 
    VGG16, VGG19,
    ResNet50, ResNet50V2,
    ResNet101, ResNet101V2,
    ResNet152, ResNet152V2,
    Xception, InceptionV3, InceptionResNetV2, 
    MobileNet, MobileNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    NASNetLarge, NASNetMobile
)
from tensorflow.keras.applications import (
    xception, vgg16, vgg19, resnet, resnet_v2,
    inception_v3, inception_resnet_v2,
    mobilenet, mobilenet_v2, densenet, nasnet
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image as pil_image


_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}
# These methods were only introduced in version 3.4.0 (2016).
if hasattr(pil_image, 'HAMMING'):
    _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
if hasattr(pil_image, 'BOX'):
    _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
# This method is new in version 1.1.3 (2013).
if hasattr(pil_image, 'LANCZOS'):
    _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(
        pil_img, min(pil_img.size), min(pil_img.size)
    )


def load_img(
    bytes_io,
    target_size=None,
    interpolation='nearest'):
    img = crop_max_square(pil_image.open(bytes_io)).convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


class Image2Vec:
    def __init__(self, model_name=None):
        if model_name == 'Xception':
            base_model = Xception(
                weights='imagenet'
            )
            self.preprocess_input = xception.preprocess_input
        elif model_name == 'VGG19':
            base_model = VGG19(
                weights='imagenet'
            )
            self.preprocess_input = vgg19.preprocess_input
        elif model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet'
            )
            self.preprocess_input = resnet.preprocess_input
        elif model_name == 'ResNet101':
            base_model = ResNet101(
                weights='imagenet'
            )
            self.preprocess_input = resnet.preprocess_input
        elif model_name == 'ResNet152':
            base_model = ResNet152(
                weights='imagenet'
            )
            self.preprocess_input = resnet.preprocess_input
        elif model_name == 'ResNet50V2':
            base_model = ResNet50V2(
                weights='imagenet'
            )
            self.preprocess_input = resnet_v2.preprocess_input
        elif model_name == 'ResNet101V2':
            base_model = ResNet101V2(
                weights='imagenet'
            )
            self.preprocess_input = resnet_v2.preprocess_input
        elif model_name == 'ResNet152V2':
            base_model = ResNet152V2(
                weights='imagenet'
            )
            self.preprocess_input = resnet_v2.preprocess_input
        elif model_name == 'InceptionV3':
            base_model = InceptionV3(
                weights='imagenet'
            )
            self.preprocess_input = inception_v3.preprocess_input
        elif model_name == 'InceptionResNetV2':
            base_model = InceptionResNetV2(
                weights='imagenet'
            )
            self.preprocess_input = inception_resnet_v2.preprocess_input
        elif model_name == 'DenseNet121':
            base_model = DenseNet121(
                weights='imagenet'
            )
            self.preprocess_input = densenet.preprocess_input
        elif model_name == 'DenseNet169':
            base_model = DenseNet169(
                weights='imagenet'
            )
            self.preprocess_input = densenet.preprocess_input
        elif model_name == 'DenseNet201':
            base_model = DenseNet201(
                weights='imagenet'
            )
            self.preprocess_input = densenet.preprocess_input
        elif model_name == 'NASNetLarge':
            base_model = NASNetLarge(
                weights='imagenet'
            )
            self.preprocess_input = nasnet.preprocess_input
        elif model_name == 'NASNetMobile':
            base_model = NASNetMobile(
                weights='imagenet'
            )
            self.preprocess_input = nasnet.preprocess_input
        elif model_name == 'MobileNet':
            base_model = MobileNet(
                weights='imagenet'
            )
            self.preprocess_input = mobilenet.preprocess_input
        elif model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet'
            )
            self.preprocess_input = mobilenet_v2.preprocess_input
        else:
            base_model = VGG16(
                weights='imagenet'
            )
            self.preprocess_input = vgg16.preprocess_input
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output
        )

    def extract_features(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        return self.model.predict(x)[0]
