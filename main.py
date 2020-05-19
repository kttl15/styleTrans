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
from firebaseUtils import FirebaseStorageUtils
from firestoreUtils import FirestoreUtils
import asyncio


# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


class Extractor(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(Extractor, self).__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
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
        style_outputs = [
            self.gram_matrix(style_output) for style_output in style_outputs
        ]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}

    def vgg_layers(self, layer_names):
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

    def gram_matrix(self, input_tensor):
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


class StyleTransfer:
    def __init__(self):
        # with open(processDict, "r") as f:
        #     processDict = json.load(f)

        # get the number of processes
        # self.processDict = processDict
        # processDictLen = 0
        # for uid in self.processDict.keys():
        #     processDictLen += len(self.processDict[uid])

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

        # getMoreData = False
        # isDoneGettingProcessDict = False
        # isGettingProcessDict = False

        # with tqdm(total=processDictLen) as pbar:
        # for uid in self.processDict.keys():
        #     for process in self.processDict[uid]:
        #         # self.run(process)
        #         time.sleep(1)
        #         processDictLen -= 1
        #         if processDictLen <= 5 and getMoreData:
        #             self.processDict = firestore.getProcessDict(returnDict=True)

        # get more processes

        # firestore.getProcessDict()

        # firestore.updateField(uid, process)
        # pbar.update(1)

    def run(self, process: dict):
        # get details from json object
        locContent = f"/home/a/Desktop/downloaded/{process['locContent']}"
        locStyle = f"/home/a/Desktop/downloaded/{process['locStyle']}"
        content_img = self.load_img(locContent)
        style_img = self.load_img(locStyle)
        epochs = int(process["epoch"])
        numberOfImageToSave = min(epochs, 4)
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
        epochsToSave = np.linspace(1, epochs, numberOfImageToSave).round().astype(int)

        img = tf.Variable(content_img)
        train_step = self.new_train_step()
        for epoch in tqdm(range(epochs)):
            for m in range(steps_per_epoch):
                train_step(
                    img,
                    total_variation_weight,
                    self.styleWeight,
                    self.contentWeight,
                    self.extractor,
                    self.opt,
                )
            if epoch + 1 in epochsToSave:
                self.tensor_to_image(img).save(
                    os.path.join(
                        "/".join(locContent.split("/")[:-1]), f"output-{epoch+1}.jpg",
                    )
                )

    def clip_0_1(self, img):
        """[summary]
        clip any value above 1.0 or below 0.0

        Arguments:
            img {tf.Tensor} -- [unclipped image]

        Returns:
            [tf.Tensor] -- [clipped image]
        """
        return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

    def tensor_to_image(self, tensor):
        """[summary]
        convert tensor representation of image back to a PIL.Image image

        Arguments:
            tensor {tf.Tensor} -- [tensor containing the image]

        Returns:
            [PIL.Image] -- [image]
        """
        tensor = tensor * 255.0
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    def load_img(self, path_to_img: str):
        """[summary]
        load image as tensor as resize

        Arguments:
            path_to_img {str} -- [path to image]

        Returns:
            [tf.Tensor] -- [tensor containing the image]
        """
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
            """[summary]
            calculating loss and applying gradients

            Arguments:
                img {tf.Tensor} -- [description]
                tvw {float} -- [description]
                sw {float} -- [description]
                cw {float} -- [description]
                extractor {[type]} -- [description]
                opt {[type]} -- [description]
            """
            with tf.GradientTape() as tape:
                outputs = extractor(img)
                loss = self.style_content_loss(outputs)
                loss += tvw * tf.image.total_variation(img)
            grad = tape.gradient(loss, img)
            opt.apply_gradients([(grad, img)])
            img.assign(self.clip_0_1(img))

        return train_step


def getMoreData():
    firestore.getProcessDict(returnDict=True)


if __name__ == "__main__":
    config = {"serviceAccount": "/home/a/Desktop/gan/serviceAccount.json"}
    firebaseStorage = FirebaseStorageUtils(config)
    firestore = FirestoreUtils(config)
    processDict = firestore.getProcessDict(returnDict=True)
    minTimeBetweenRequests = 300
    style = StyleTransfer()

    while True:
        processDictLen = 0
        restart = False
        lastRequest = time.time()

        for uid in processDict.keys():
            processDictLen += len(processDict[uid])

        print(f"data len: {processDictLen}")
        if processDictLen <= 1:
            if len(processDict[list(processDict.keys())[0]]) == 0:
                print("No More Data. Breaking.")
                break

        for uid in processDict.keys():
            for process in processDict[uid]:
                # style.run(process=process)
                time.sleep(1)
                # firestore.updateField(uid, process["processName"])
                processDictLen -= 1
                if processDictLen <= 5:
                    if time.time() - lastRequest >= minTimeBetweenRequests:
                        print("Getting More Data.")
                        processDict = firestore.getProcessDict(returnDict=True)
                        print("Restarting.")
                        restart = True
                    elif processDictLen <= 0:
                        print(
                            f"Last Request Within 5 Minutes. Sleeping for approx {lastRequest + minTimeBetweenRequests - time.time()}s"
                        )
                        time.sleep(1)
                    else:
                        print("Continue")

                if restart:
                    break
            if restart:
                break


# TODO: find a way to control how much colour is used
