import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import src.utils as utils
import src.models.model_utils as mutils


class ModelMaker:
    def __init__(self, src_dir, dst_dir, est_file, cls_file,
                 info_file, graph_file, hist_file, ft_hist_file,
                 input_size, dense_dims, lr, ft_lr, min_lr, min_ft_lr,
                 batch_size, reuse_count, epochs, valid_rate,
                 es_patience, lr_patience, ft_start):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.est_file = est_file
        self.cls_file = cls_file
        self.info_file = info_file
        self.graph_file = graph_file
        self.hist_file = hist_file
        self.ft_hist_file = ft_hist_file
        self.input_size = input_size
        self.dense_dims = dense_dims
        self.lr = lr
        self.ft_lr = ft_lr
        self.min_lr = min_lr
        self.min_ft_lr = min_ft_lr
        self.batch_size = batch_size
        self.reuse_count = reuse_count
        self.epochs = epochs
        self.valid_rate = valid_rate
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.ft_start = ft_start

    def define_model(self):
        '''VGG16モデルの定義
        各layerを凍結して、出力層のみ所定のクラス数に対応させる
        '''
        base_model = VGG16(include_top=False,
                           input_shape=(*self.input_size, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output

        x = Flatten()(x)

        for dim in self.dense_dims[:-1]:
            x = mutils.add_dense_layer(x, dim)

        x = mutils.add_dense_layer(
            x, self.dense_dims[-1], use_bn=False, activation='softmax'
        )

        model = Model(base_model.input, x)

        model.compile(
            optimizer=Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def unfreeze_layers(self, model):
        '''ファインチューニング用に、特定のlayerをトレーニング可能にする
        '''
        for layer in model.layers[self.ft_start:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(lr=self.ft_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit_model(self):
        '''モデルをトレーニングする
        以下の二つのステップを含む
        1.転移学習
        2.ファインチューニング
        '''
        train_ds, train_n, valid_ds, valid_n, cls_info = utils.make_generator(
            self.src_dir, self.valid_rate, self.input_size, self.batch_size
        )

        model = self.define_model()

        early_stopping = EarlyStopping(
            patience=self.es_patience,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr_op = ReduceLROnPlateau(
            patience=self.lr_patience,
            min_lr=self.min_lr,
            verbose=1
        )
        callbacks = [early_stopping, reduce_lr_op]

        # 1回目は転移学習
        history = model.fit(
            train_ds,
            steps_per_epoch=int(
                # reuse_countは1epochにおけるAugumentationの繰り返し回数
                train_n * self.reuse_count / self.batch_size
            ),
            epochs=self.epochs,
            validation_data=valid_ds,
            validation_steps=int(
                valid_n * self.reuse_count / self.batch_size
            ),
            callbacks=callbacks
        )

        self.unfreeze_layers(model)

        # reduce_lr_opは上書きしているため注意
        reduce_lr_op = ReduceLROnPlateau(
            patience=self.lr_patience,
            min_lr=self.min_ft_lr,
            verbose=1
        )
        callbacks = [early_stopping, reduce_lr_op]

        # 2回目はファインチューニング
        fit_history = model.fit(
            train_ds,
            steps_per_epoch=int(
                train_n * self.reuse_count / self.batch_size
            ),
            epochs=self.epochs,
            validation_data=valid_ds,
            validation_steps=int(
                valid_n * self.reuse_count / self.batch_size
            ),
            callbacks=callbacks
        )

        return model, cls_info, history.history, fit_history.history

    # executeをtrain_modelに渡すか?
    def execute(self):
        model, cls_info, history, ft_history = self.fit_model()

        utils.mkdir(self.dst_dir, rm=True)
        model.save(self.est_file)

        mutils.save_model_info(self.info_file, self.graph_file, model)

        with open(self.cls_file, 'wb') as f:
            pickle.dump(cls_info, f)
        print(f'Classes: {cls_info}')

        utils.plot(history, self.hist_file)
        utils.plot(ft_history, self.ft_hist_file)

        def get_min(loss):
            min_val = min(loss)
            min_ind = loss.index(min_val)
            return min_val, min_ind

        print('Before fine-tuning')
        min_val, min_ind = get_min(history['val_loss'])
        print(f'val_loss: {min_val} (Epochs: {min_ind + 1})')

        print('After fine-tuning')
        min_val, min_ind = get_min(ft_history['val_loss'])
        print(f'val_loss: {min_val} (Epochs: {min_ind + 1})')
