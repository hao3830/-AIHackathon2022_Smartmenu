from .utils import load_vietocr_config_from_file
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pathlib import Path


class VietOCRPreditor:
    def __init__(self, weights=None, cfg_file="./cfg/vgg-seq2seq.yaml"):
        if isinstance(cfg_file, str):
            cfg_file = Path(cfg_file)
        if isinstance(weights, str):
            weights = Path(weights)

        # config = Cfg.load_config_from_name("vgg_seq2seq")
        config = load_vietocr_config_from_file(cfg_file)

        if weights:
            config["weights"] = str(weights)

        config["cnn"]["pretrained"] = False
        config["device"] = "cpu"
        config["predictor"]["beamsearch"] = False

        self.detector = Predictor(config)

    def __call__(self, img, batch_size=4):
        """Run OCR on image

        Args:
            img (PIL.Image | List<PIL.Image>):

        Returns:
            text
        """
        if batch_size > 1:
            return self.detector.predict_batch(img)
        else:
            return self.detector.predict(img)
