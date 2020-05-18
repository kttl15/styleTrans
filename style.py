import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
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


def clip_0_1(img):
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, style_weight, content_weight):
    """[summary]
    define the loss function

    Arguments:
        outputs {[type]} -- [description]
        style_weight {[type]} -- [description]
        content_weight {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    style_outputs = outputs["style"]
    content_outputs = outputs["content"]

    # element-wise addition of the means of the difference between the style outputs and targets
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )

    # taking into account weight
    style_loss *= style_weight / len(style_layers)

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )

    content_loss *= content_weight / len(content_layers)

    loss = style_loss + content_loss
    return loss


def new_train_step():
    @tf.function()
    def train_step(img, tvw, sw, content_weight):
        with tf.GradientTape() as tape:
            outputs = extractor(img)
            loss = style_content_loss(outputs, sw, content_weight)
            loss += tvw * tf.image.total_variation(img)
        grad = tape.gradient(loss, img)
        opt.apply_gradients([(grad, img)])
        img.assign(clip_0_1(img))
        # train_loss(loss)

    return train_step


if __name__ == "__main__":

    content_layers = ["block5_conv1", "block5_conv2", "block5_conv3", "block5_conv4"]
    style_layers = [
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

    date = str(datetime.now().strftime("%Y-%m-%d"))
    path_to_testing = "/home/a/Desktop/gan/style_results/"
    testing_files = sorted([path_to_testing + f for f in os.listdir(path_to_testing)])
    with tqdm(total=len(testing_files) * 3000 * 2 * 2) as pbar:
        for path_to_file in testing_files:
            if not date in os.listdir(path_to_file):
                os.mkdir(os.path.join(path_to_file, date))

            content_path = os.path.join(path_to_file, "base.jpg")
            style_path = os.path.join(path_to_file, "style.jpg")

            content_img = load_img(content_path)
            style_img = load_img(style_path)

            # TODO: find a way to control how much colour is used

            lr = 1e-2
            b1 = 0.99
            ep = 1e-2
            decay = 1e-8
            total_variation_weight = [100, 1000]
            style_weight = [1e-2, 1]
            content_weight = 1e3
            epochs = 30
            steps_per_epoch = 100

            for tvw in total_variation_weight:
                for sw in style_weight:

                    opt = tf.optimizers.Nadam(
                        learning_rate=lr, beta_1=b1, epsilon=ep, decay=decay,
                    )
                    new_folder = f"{datetime.now().strftime('%H:%M:%S')}_lr-{lr}_b1-{b1}_ep-{ep}_d-{decay}_tvw-{tvw}_sw-{sw}_cw-{content_weight}"
                    if not new_folder in os.listdir(os.path.join(path_to_file, date)):
                        os.mkdir(os.path.join(path_to_file, date, new_folder))

                    # write layers used to file
                    with open(
                        os.path.join(path_to_file, date, new_folder, "options.txt"), "w"
                    ) as f:
                        f.write(f'opt = {str(opt).split(".")[-1].split(" ")[0]} \n\n')
                        f.write("content_layers\n\n")
                        for c in content_layers:
                            f.write(c + "\n")
                        f.write("\nstyle_layers\n\n")
                        for s in style_layers:
                            f.write(s + "\n")
                        f.close()

                    # tensorboard logging
                    # log_dir = results_path + new_folder + '/logs'
                    # summary_writer = tf.summary.create_file_writer(log_dir)
                    # train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

                    extractor = Extractor(style_layers, content_layers)

                    style_targets = extractor(style_img)["style"]
                    content_targets = extractor(content_img)["content"]

                    img = tf.Variable(content_img)
                    train_step = new_train_step()

                    counter = 1
                    for n in range(epochs):
                        for m in range(steps_per_epoch):
                            train_step(img, tvw, sw, content_weight)
                            pbar.update(1)
                            # with summary_writer.as_default():
                            #     tf.summary.scalar('loss', train_loss.result(), step = counter)
                            # summary_writer.flush()
                            # train_loss.reset_states()
                            counter += 1

                        tensor_to_image(img).save(
                            os.path.join(
                                path_to_file, date, new_folder, f"epoch-{n+1}.jpg"
                            )
                        )
                break
            break
