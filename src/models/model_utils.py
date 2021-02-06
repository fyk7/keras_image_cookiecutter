from tensorflow.keras.layers import Activation, BatchNormalization, Dense
from tensorflow.keras.utils import plot_model


def add_dense_layer(x, dim, use_bn=True, activation='relu'):
    '''全結合層の定型処理をラッピング(Dense, Activation, BatchNormalization)
    '''
    x = Dense(dim, use_bias=not use_bn)(x)
    x = Activation(activation)(x)
    if use_bn:
        x = BatchNormalization()(x)
    return x


def save_model_info(info_file, graph_file, model):
    '''訓練済みモデルの構造を画像ファイルに保存
    '''
    with open(info_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '¥n'))
    # you should install pydot and graphviz through pip
    # you should install gtaphviz to your local OS (e.g. brew install graphviz)
    plot_model(model, to_file=graph_file, show_shapes=True)
