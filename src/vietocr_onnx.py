from re import I
import yaml
import torch
import numpy as np
import math
from PIL import Image
import onnxruntime
from pathlib import Path


class Vocab:
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c: i + 4 for i, c in enumerate(chars)}

        self.i2c = {i + 4: c for i, c in enumerate(chars)}

        self.i2c[0] = "<pad>"
        self.i2c[1] = "<sos>"
        self.i2c[2] = "<eos>"
        self.i2c[3] = "*"

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = "".join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    cnn_session, encoder_session, decoder_session = session
    # create cnn input
    cnn_input = {cnn_session.get_inputs()[0].name: img}
    src = cnn_session.run(None, cnn_input)

    # create encoder input
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)

    translated_sentence = [[sos_token] * len(img)]
    max_length = 0

    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {
            decoder_session.get_inputs()[0].name: tgt_inp[-1],
            decoder_session.get_inputs()[1].name: hidden,
            decoder_session.get_inputs()[2].name: encoder_outputs,
        }

        output, hidden, _ = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence


def load_config(config_file):
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname, base_file="./cfg/base.yaml"):
        base_config = load_config(base_file)

        with open(fname, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    def save(self, fname):
        with open(fname, "w") as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)


class Vocab:
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c: i + 4 for i, c in enumerate(chars)}

        self.i2c = {i + 4: c for i, c in enumerate(chars)}

        self.i2c[0] = "<pad>"
        self.i2c[1] = "<sos>"
        self.i2c[2] = "<eos>"
        self.i2c[3] = "*"

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = "".join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 4

    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert("RGB")

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


class VietOCROnnx:
    def __init__(self, weights="vietocr_weights", cfg_file="vietocr_weights"):
        if isinstance(cfg_file, str):
            cfg_file = Path(cfg_file)
        if isinstance(weights, str):
            weights = Path(weights)

        config = Cfg.load_config_from_file(cfg_file)
        config["cnn"]["pretrained"] = False
        config["device"] = "cpu"
        vocab = Vocab(config["vocab"])

        cuda = False
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )

        cnn_session = onnxruntime.InferenceSession(
            str(weights / "cnn.onnx"), providers=providers
        )
        encoder_session = onnxruntime.InferenceSession(
            str(weights / "encoder.onnx"), providers=providers
        )
        decoder_session = onnxruntime.InferenceSession(
            str(weights / "decoder.onnx"), providers=providers
        )

        self.session = (cnn_session, encoder_session, decoder_session)
        self.config = config
        self.vocab = vocab

    def __call__(self, img, batch_size=4):
        """Run OCR on image

        Args:
            img (PIL.Image | List<PIL.Image>):

        Returns:
            text
        """
        if isinstance(img, list) and batch_size > 1:

            def _f(im):
                im = process_input(
                    im,
                    self.config["dataset"]["image_height"],
                    self.config["dataset"]["image_min_width"],
                    self.config["dataset"]["image_max_width"],
                )
                im = im.cpu().numpy()
                return im

            _img = list(map(_f, img))

            bucket = {}
            for im in _img:
                k = im.shape[3]
                if k not in bucket:
                    bucket[k] = []
                bucket[k].append(im)

            texts = []
            for _, v in bucket.items():
                for i in range(math.ceil(len(v) / batch_size)):
                    _v = v[i * batch_size : (i + 1) * batch_size]

                    orin_size = len(_v)

                    # Ensure batch of v always equal to batch_size
                    if orin_size < batch_size:
                        _v = _v + [_v[-1]] * (batch_size - orin_size)

                    _v = np.vstack(_v)
                    _texts = translate_onnx(_v, self.session)
                    for j in range(orin_size):
                        t = self.vocab.decode(_texts[j].tolist())
                        texts.append(t)

            return texts
        else:
            _img = process_input(
                img,
                self.config["dataset"]["image_height"],
                self.config["dataset"]["image_min_width"],
                self.config["dataset"]["image_max_width"],
            )
            _img = _img.cpu().numpy()

            text = translate_onnx(_img, self.session)[0].tolist()
            text = self.vocab.decode(text)
            return text
