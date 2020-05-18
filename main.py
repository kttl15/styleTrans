import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import json
import pathlib
import matplotlib.pyplot as plt
from datetime import datetime
import time
import downloader
import test


gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


def tensor_to_image(tensor):
    tensor = tensor * 255.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def vgg_layers(layer_names):
    """[summary]
    creates a vgg model that returns a list of intermediate output values

    Arguments:
        layer_names {[String]} -- [list of layers for content and style]

    Returns:
        [tf.keras.Model] -- [a vgg model]
    """

    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    new_vgg = tf.keras.Model([vgg.input], outputs)
    return new_vgg


def gram_matrix(input_tensor):
    """[summary]
    style of an image can be described by the means and correlations across the different feature maps.
    calculate a gram matrix that includes this information by taking the outer product of the feature vetors with itself
    at each location and averaging that outer product over all locations.

    G_{c, d}^l = sum_{i,j} * F_{i,j,c}^l (x) * F_{i,j,d}^l (x) / IJ

    Inputs:
        input_tensor [tf.Tensor] -- [
            rank 4 tensor -- [1, i, j, c]
        ]

    Returns:
        result [tf.tensor] -- [
            rank 3 tensor -- [1, n, n]  ex: (1, 128, 128)
        ]
    """

    # F_{i,j,c}^l (x) * F_{i,j,d}^l (x)
    result = tf.linalg.einsum("lijc,lijd->lcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    IJ = input_shape[1] * input_shape[2]  # i * j = IJ
    IJ = tf.cast(IJ, tf.float32)  # convert to tf.float32
    result = result / IJ
    return result


class Extractor(tf.keras.models.Model):
    def __init__(self, style_layers, content_layer):
        super(Extractor, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        """[summary]

        Arguments:
            inputs {[float]} -- [between 0 and 1]

        Returns:
            dict of content and style outputs
        """

        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


class StyleTransfer:
    def __init__(self, processDict: str):
        with open(processDict, "r") as f:
            processDict = json.load(f)

        # get the number of processes
        processDictLen = 0
        for uid in processDict.keys():
            processDictLen += len(processDict[uid])
        self.content_layers = [
            "block5_conv1",
            "block5_conv2",
            "block5_conv3",
            "block5_conv4",
        ]

        self.style_layers = [
            "block2_conv1",
            "block2_conv2",
            "block3_conv1",
            "block3_conv2",
            "block3_conv3",
            "block3_conv4",
            # 'block4_conv1',
            # 'block4_conv2',
            # 'block4_conv3',
            # 'block4_conv4',
        ]
        with tqdm(total=processDictLen) as pbar:
            for uid in processDict.keys():
                for process in processDict[uid]:
                    self.run(process)
                    pbar.update(1)

    def run(self, process):
        # get details from json object
        locContent = f"downloaded/{process['locContent']}"
        locStyle = f"downloaded/{process['locStyle']}"
        content_img = load_img(locContent)
        style_img = load_img(locStyle)
        epochs = int(process["epoch"])
        numberOfImageToSave = 4
        self.contentWeight = int(process["contentWeight"])
        self.contentWeight = 1e3 * self.contentWeight ** 2 * 0.5
        self.styleWeight = int(process["styleWeight"])
        self.styleWeight = -1.1e-3 * self.styleWeight + 0.0111
        lr = 1e-2
        b1 = 0.99
        ep = 1e-2
        decay = 1e-8

        self.opt = tf.optimizers.Nadam(
            learning_rate=lr, beta_1=b1, epsilon=ep, decay=decay,
        )
        self.extractor = Extractor(self.style_layers, self.content_layers)
        self.style_targets = self.extractor(style_img)["style"]
        self.content_targets = self.extractor(content_img)["content"]
        total_variation_weight = 1000
        steps_per_epoch = 100
        epochsToSave = np.linspace(1, epochs, numberOfImageToSave).round()

        img = tf.Variable(content_img)
        train_step = self.new_train_step()

        for epoch in range(epochs):
            for m in range(steps_per_epoch):
                self.train_step(
                    img,
                    total_variation_weight,
                    self.styleWeight,
                    self.contentWeight,
                    self.extractor,
                    self.opt,
                )
            if epoch in epochsToSave:
                tensor_to_image(img).save(
                    os.path.join(
                        "/".join(locContent.split("/")[:-1]), f"output-{epoch+1}.jpg",
                    )
                )

    def clip_0_1(self, img):
        return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

    def style_content_loss(self, outputs):
        """[summary]
        define the loss function

        Arguments:
            outputs {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        # element-wise addition of the means of the difference between the style outputs and targets
        style_loss = tf.add_n(
            [
                tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                for name in style_outputs.keys()
            ]
        )

        # taking into account weight
        style_loss *= self.styleWeight / len(self.style_layers)

        content_loss = tf.add_n(
            [
                tf.reduce_mean(
                    (content_outputs[name] - self.content_targets[name]) ** 2
                )
                for name in content_outputs.keys()
            ]
        )

        content_loss *= self.contentWeight / len(self.content_layers)

        loss = style_loss + content_loss
        return loss

    def new_train_step(self):
        @tf.function()
        def train_step(img, tvw, sw, cw, extractor, opt):
            with tf.GradientTape() as tape:
                outputs = self.extractor(img)
                loss = self.style_content_loss(outputs, sw, cw)
                loss += tvw * tf.image.total_variation(img)
            grad = tape.gradient(loss, img)
            self.opt.apply_gradients([(grad, img)])
            img.assign(self.clip_0_1(img))

        return train_step


# def main(
#     locContent: str, locStyle: str, epochs: int, contentWeight: int, styleWeight: int
# ):


# if __name__ == "__main__":


# TODO: find a way to control how much colour is used

