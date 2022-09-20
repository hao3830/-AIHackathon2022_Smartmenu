from .yolo_onnx import YOLOOnnx
from .vietocr_onnx import VietOCROnnx
from .symspell import SymSpellWrapper
from .vietocr import VietOCRPreditor
from .yolo import Yolo
import torch

from PIL import Image


def xyxy24points(xyxy):
    return [
        (xyxy[0], xyxy[1]),
        (xyxy[2], xyxy[1]),
        (xyxy[2], xyxy[3]),
        (xyxy[0], xyxy[3]),
    ]


class Reader:
    def __init__(
        self,
        detector_weights,
        recognizer_weights,
        vietocr_cfg="./cfg/vgg-seq2seq.yaml",
        batch_size=4,
        ocr_correct=True,
        pt=False,
    ):
        # Prepaire for YOLO
        if pt:
            self.detector = Yolo(detector_weights)
        else:
            self.detector = YOLOOnnx(detector_weights)

        # Prepare for VietOCR
        if pt:
            self.recognizer = VietOCRPreditor(
                cfg_file=vietocr_cfg,
            )
        else:
            self.recognizer = VietOCROnnx(
                weights=recognizer_weights,
                cfg_file=vietocr_cfg,
            )

        self.symspell = None
        if ocr_correct:
            self.symspell = SymSpellWrapper()

        self.batch_size = batch_size

    def __call__(self, img0):
        """Run detect and recognize on image

        Args:
            img0 (cv2 img or np.array):
        """
        boxes = self.detector(img0)

        results = []

        batch = []
        for i, box in enumerate(boxes):
            for *xyxy, conf, clsId in reversed(box):
                xyxy = list(map(round, torch.tensor(xyxy).tolist()))

                cropped = img0[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
                cropped = Image.fromarray(cropped)

                batch.append(cropped)

                results.append(
                    [
                        xyxy24points(xyxy),
                        "",
                        conf.cpu().item(),
                    ]
                )

        def _f(t):
            t = t.lower()

            if "xia" in t:
                t = t.replace("xia", "xìa")

            if self.symspell:
                _t = self.symspell.correct(t)
                # if t != _t:
                # print(t, _t)
                return _t
            return t

        if self.batch_size > 1:
            texts = self.recognizer(batch, batch_size=self.batch_size)
            for i, t in enumerate(texts):
                results[i][1] = _f(t)
        else:
            for i, cropped in enumerate(batch):
                text = self.recognizer(cropped, batch_size=1)
                results[i][1] = _f(text)

        return results
