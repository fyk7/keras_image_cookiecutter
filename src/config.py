import os
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]

TRN_SRC_DIR = os.path.join(project_dir, 'data/processed/stationery/training')

TRN_DST_DIR = os.path.join(project_dir, 'models')
TRN_EST_FILE = os.path.join(TRN_DST_DIR, 'estimator.h5')
TRN_CLS_FILE = os.path.join(TRN_DST_DIR, 'class.pkl')

TRN_RESULT_DIR = os.path.join(project_dir, 'reports/train_results')
TRN_INFO_FILE = os.path.join(TRN_DST_DIR, 'model_info.txt')
TRN_GRAPH_FILE = os.path.join(TRN_DST_DIR, 'model_graph.txt')
TRN_HIST_FILE = os.path.join(TRN_DST_DIR, 'history.pdf')
TRN_FT_HIST_FILE = os.path.join(TRN_DST_DIR, 'ft_history.pdf')

TRN_INPUT_SIZE = (160, 160)
TRN_DENSE_DIMS = [4096, 2048, 1024, 128]
TRN_LR = 1e-4
TRN_FT_LR = 1e-5
TRN_MIN_LR = 1e-7
TRN_MIN_FT_LR = 1e-8
TRN_BATCH_SIZE = 32
TRN_REUSE_CNT = 10
TRN_EPOCHS = 200
# for easily check
# TRN_REUSE_CNT = 1
# TRN_EPOCHS = 1
TRN_VALID_RATE = 0.2
TRN_ES_PATIENCE = 30
TRN_LR_PATIENCE = 10
TRN_FT_START = 15


# TODO decide estimator destination path
EST_SRC_DIR = os.path.join(project_dir, 'data/processed/stationery/test')
EST_DST_DIR = os.path.join(project_dir, 'reports/test_results')
EST_DRS_FILE = os.path.join(EST_DST_DIR, 'detailed_result.txt')
EST_SRS_FILE = os.path.join(EST_DST_DIR, 'summary_result.txt')
