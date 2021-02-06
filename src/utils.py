import os
import sys
import shutil
import logging


import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np


def mkdir(d, rm=False):
    if rm:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    else:
        try:
            os.makedirs(d)
        except FileExistsError:
            pass


# TODO 更に適切な置き場所があるかもしれない(e.g. src/data/utils)
def make_generator(src_dir, valid_rate, input_size, batch_size):
    '''Dataset generatorを作成する関数
    dir -> generator -> Datasetの流れでデータセットを作成
    src_dir下のディレクトリ名が自動でクラス名となる(flow_from_directory)
    ImageDataGeneratorのパラメータはターゲットにより柔軟に変更(道路標識の上下フリップは必要ないetc...)
    '''
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=30,
        zoom_range=[0.7, 1.3],
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=valid_rate
    )

    # directoryの構造・名称から自動でdata_generatorを作成
    train_generator = train_datagen.flow_from_directory(
        src_dir,
        target_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='training'
    )

    valid_generator = train_datagen.flow_from_directory(
        src_dir,
        target_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='validation'
    )

    train_ds = Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [None, *train_generator.image_shape],
            [None, train_generator.num_classes]
        )
    )

    valid_ds = Dataset.from_generator(
        lambda: valid_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [None, *valid_generator.image_shape],
            [None, valid_generator.num_classes]
        )
    )

    train_ds = train_ds.repeat()
    valid_ds = valid_ds.repeat()

    cls_info = {v: k for k, v in train_generator.class_indices.items()}

    return train_ds, train_generator.n, valid_ds, valid_generator.n, cls_info


# TODO visualize.pyに移動した方が良い可能性がある
def plot(history, filename):
    '''loss, accuracyを可視化し, 特定のファイルに保存する関数
    '''
    def add_subplot(nrows, ncols, index, xdata, train_ydata, valid_ydata, ylim, ylabel):
        plt.subplot(nrows, ncols, index)
        plt.plot(xdata, train_ydata, label='training', linestyle='--')
        plt.plot(xdata, valid_ydata, label='training')
        plt.xlim(1, len(xdata))
        plt.ylim(*ylim)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

    plt.figure(fig_size=(10, 10))
    xdata = range(1, 1 + len(history['loss']))
    add_subplot(2, 1, 1, xdata, history['loss'],
                history['val_loss'], (0, 5), 'loss')
    add_subplot(2, 1, 2, xdata, history['accuracy'],
                history['val_accuracy'], (0, 5), 'accuracy')
    plt.savefig(filename)
    plt.close('all')


def load_target_image(filename, input_size):
    '''推定対象のイメージを読み込む関数
    '''
    # PIL形式、modelにインプットできるようにresize
    img = load_img(filename, target_size=input_size)
    img = img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    return img


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(stderr_level)
        logger.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(formatter)
        handler.setLevel(file_level)
        logger.addHandler(handler)

    logger.info("logger set up")

    # TODO logの出力先を適切に定義
    # if not os.path.isdir('./logs'):
    #     os.makedirs('./logs')

    return logger
