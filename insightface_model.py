import argparse
import os
import os.path as osp
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

import onnxruntime


def model_insightface():
    onnxruntime.set_default_logger_severity(3)

    assets_dir = osp.expanduser('/home/hungnm/Downloads/face/face_recognition/buffalo_l')

    ap = argparse.ArgumentParser()

    ap.add_argument("--retrain", default=False,
                    help="True to retrain")

    args = vars(ap.parse_args())

    # Detector = mtcnn_detector
    detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
    detector.prepare(0)
    model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
    rec = ArcFaceONNX(model_path)
    rec.prepare(0)
    retrain = bool(args["retrain"])
    return rec, retrain, detector