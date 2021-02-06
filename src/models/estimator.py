import os
import pickle

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import src.utils as utils


class Estimator:
    def __init__(
            self, src_dir, dst_dir,
            est_file, cls_file, drs_file, srs_file, input_size):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.est_file = est_file
        self.cls_file = cls_file
        self.drs_file = drs_file
        self.srs_file = srs_file
        self.input_size = input_size

    def execute(self):
        estimator = load_model(self.est_file)

        with open(self.cls_file, 'rb') as f:
            cls_info = pickle.load(f)

        pred_labels, true_labels, output = [], [], []

        for subdir in os.listdir(self.src_dir):
            for f in os.listdir(os.path.join(self.src_dir, subdir)):
                filename = os.path.join(self.src_dir, subdir, f)
                img = utils.load_img(filename, self.input_size)
                pred_class = np.argmax(estimator.predict(img))
                pred_label = cls_info[pred_class]
                pred_labels.append(pred_label)

                true_label = subdir
                true_labels.append(true_label)

                output.append(f'{filename} -> {pred_label}')

        report = classificaiton_report(true_label, pred_labels)
        labels = list(cls_info.values())
        cnfmtx = confusion_matrix(true_labels, pred_labels, labels)
        cm = pd.DataFrame(cnfmtx, index=labels, columns=labels)

        utils.mkdir(self.dst_dir, rm=True)
        with open(self.drs_file, 'w') as f:
            f.writelines(output)

        with open(self.srs_file, 'w') as f:
            f.write(report)
            f.write('¥n¥n')
            f.write(str(cm))
            f.write('¥n')
