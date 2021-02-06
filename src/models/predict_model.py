import os
import pickle
import pandas as pd
import datetime

from src import config
from src.utils import setup_logger
from src.models.estimator import Estimator


def predict():
    estimator = Estimator(
        src_dir=config.EST_SRC_DIR,
        dst_dir=config.EST_DST_DIR,
        est_file=config.TRN_EST_FILE,
        cls_file=config.TRN_CLS_FILE,
        drs_file=config.EST_DRS_FILE,
        srs_file=config.EST_SRS_FILE,
        input_size=config.TRN_INPUT_SIZE
    )
    estimator.execute()


if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # TODO loggerの出力先の設定
    # logger = setup_logger('./logs/predict_{0}.log'.format(NOW))
    logger = setup_logger()
    logger.info('Prediction step starts')
    # TODO Estimatorの引数をlogファイルに出力する

    predict()
