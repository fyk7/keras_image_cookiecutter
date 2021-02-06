import os
import pickle
import pandas as pd
import datetime

from src import config
from src.utils import setup_logger
from src.models.model_maker import ModelMaker


def train():
    n_class = len(os.listdir(config.TRN_SRC_DIR))
    # モデルの最終出力層にクラス数を追加
    config.TRN_DENSE_DIMS.append(n_class)

    maker = ModelMaker(
        src_dir=config.TRN_SRC_DIR,
        dst_dir=config.TRN_DST_DIR,
        est_file=config.TRN_EST_FILE,
        cls_file=config.TRN_CLS_FILE,
        info_file=config.TRN_INFO_FILE,
        graph_file=config.TRN_GRAPH_FILE,
        hist_file=config.TRN_HIST_FILE,
        ft_hist_file=config.TRN_FT_HIST_FILE,
        input_size=config.TRN_INPUT_SIZE,
        dense_dims=config.TRN_DENSE_DIMS,
        lr=config.TRN_LR,
        ft_lr=config.TRN_FT_LR,
        min_lr=config.TRN_MIN_LR,
        min_ft_lr=config.TRN_MIN_FT_LR,
        batch_size=config.TRN_BATCH_SIZE,
        reuse_count=config.TRN_REUSE_CNT,
        epochs=config.TRN_EPOCHS,
        valid_rate=config.TRN_VALID_RATE,
        es_patience=config.TRN_ES_PATIENCE,
        lr_patience=config.TRN_LR_PATIENCE,
        ft_start=config.TRN_FT_START
    )
    maker.execute()


if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # TODO loggerの出力先の設定
    # logger = setup_logger('./logs/train_{0}.log'.format(NOW))
    logger = setup_logger()
    logger.info('Training step starts')
    # TODO ModelMakerの引数をlogファイルに出力する

    train()
