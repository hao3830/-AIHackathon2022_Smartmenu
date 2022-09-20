def warmup_easyocr():
    import easyocr

    model = easyocr.Reader(["vi", "en"])


def warmup_vietocr():
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    from src.utils import load_vietocr_config_from_file

    # config = Cfg.load_config_from_name("vgg_seq2seq")
    config = load_vietocr_config_from_file("./cfg/vgg-seq2seq.yaml")
    config["cnn"]["pretrained"] = False
    config["device"] = "cpu"
    config["predictor"]["beamsearch"] = False

    detector = Predictor(config)


def main():
    warmup_vietocr()


if __name__ == "__main__":
    main()
