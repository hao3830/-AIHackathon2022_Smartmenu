# import easyocr
from .text_reader import Reader
from .utils import extract_pairs, clean_price, load_reference_csv
from .translator import Translator
from pathlib import Path
from fuzzywuzzy import process, fuzz


class Extractor:
    def __init__(
        self,
        detector_weights,
        recognizer_weights,
        vietocr_cfg="./cfg/vgg-seq2seq.yaml",
        referenced_csv="./src/label.csv",
        batch_size=4,
        pt=False,
    ):
        if isinstance(referenced_csv, str):
            referenced_csv = Path(referenced_csv)

        referenced_data = load_reference_csv(referenced_csv)
        self.referenced_data = referenced_data
        self.referenced_data_keys = list(referenced_data.keys())

        self.reader = Reader(
            detector_weights=detector_weights,
            recognizer_weights=recognizer_weights,
            vietocr_cfg=vietocr_cfg,
            batch_size=batch_size,
            pt=pt,
        )
        # self._reader = easyocr.Reader(["en", "vi"])
        # self.reader = lambda x: self._reader.readtext(x)

        self.translator = Translator(referenced_csv)
        self.batch_size = batch_size

    def extract_menu(self, img):
        boxes = self.reader(img)
        pairs = extract_pairs(boxes)

        thres = 85
        thres_2 = 90
        diff_thres = 0.15
        correct_price_thres = 30000
        correct_group_price_thres = 30000

        unique_names = {}

        # pair: VNName, price, ENName
        _pairs = []
        for p in pairs:
            vi_name = p.name.upper()
            en_name = ""

            price = clean_price(str(p.price))
            _orin_price = price

            def _remove_postfix(x):
                name, percent = x
                if "_" not in name:
                    return x
                name = name.split("_")[0]
                return name, percent

            # process case with multiple drink size
            _t = process.extract(
                vi_name.lower(),
                self.referenced_data_keys,
                limit=5,
            )

            # print(vi_name, _t, price)

            def diff_ratio(text1, text2):
                diff = abs(len(text1) - len(text2))
                m = max(len(text1), len(text2))
                return diff / m

            # Process case predicted drink very similar to referenced CSV with multiple sizes S M L
            if "COMBO" in vi_name:
                s = vi_name.split("COMBO")
                vi_name = " COMBO ".join(s)

                _t2 = list(filter(lambda x: x[1] >= thres, _t))
                _t2 = list(map(_remove_postfix, _t2))
                _t2 = list(map(lambda x: self.referenced_data[x[0]], _t))

                d = _t2[0]
                vi_name = d[1]
                en_name = d[2]
            elif (
                _t[0][1] >= thres
                and diff_ratio(vi_name, _t[0][0]) < diff_thres
                and _t[0][0][-1] in "sml"
            ):
                _t2 = list(filter(lambda x: x[0][-1] in "sml", _t))
                _t2 = list(filter(lambda x: x[1] >= thres, _t2))
                _t2 = list(map(_remove_postfix, _t2))
                _t2 = list(map(lambda x: self.referenced_data[x[0]], _t2))

                rat = fuzz.ratio(_t2[0][1].lower(), vi_name.lower())
                if rat != 100 and rat > thres_2:
                    _t2 = sorted(
                        _t2,
                        key=lambda x: 0
                        if x[-1] == "NOT GIVEN"
                        else abs(int(x[-1]) - price),
                    )

                _t3 = _t2[0]
                vi_name = _t3[1]
                en_name = _t3[2]

                if _t3[3] != "NOT GIVEN" and abs(int(_t3[3]) - price) > 30000:
                    price = int(_t3[3])

            # Process case that predicted very similar to referenced CSV
            elif _t[0][1] >= thres and diff_ratio(vi_name, _t[0][0]) < diff_thres:
                _t2 = list(filter(lambda x: x[1] >= thres, _t))
                _t2 = list(map(_remove_postfix, _t2))
                _t2 = list(map(lambda x: self.referenced_data[x[0]], _t))

                rat = fuzz.ratio(_t2[0][1].lower(), vi_name.lower())
                if rat != 100 and rat > thres_2:
                    _t2 = sorted(
                        _t2,
                        key=lambda x: 0
                        if x[-1] == "NOT GIVEN"
                        else abs(int(x[-1]) - price),
                    )

                d = _t2[0]
                vi_name = d[1]
                en_name = d[2]

                if d[3] != "NOT GIVEN" and abs(int(d[3]) - price) > correct_price_thres:
                    price = int(d[3])

            else:
                # en_name = self.translator(vi_name)
                en_name = ""

            vi_name = vi_name.upper()
            _pairs.append(
                {
                    "vi_name": vi_name,
                    "en_name": en_name.upper(),
                    "price": price if price > 0 else "Free",
                    "orin_price": _orin_price,
                }
            )

            if vi_name not in unique_names:
                unique_names[vi_name] = len(_pairs) - 1
            else:
                _pairs[-1]["price"] = _pairs[-1]["orin_price"]
                _t = unique_names[vi_name]
                _pairs[_t]["price"] = _pairs[_t]["orin_price"]
                unique_names[vi_name] = len(_pairs) - 1

        if len(_pairs) > 3:
            for i, p in enumerate(_pairs[1:-1], start=1):
                prev = _pairs[i - 1]["price"]
                _next = _pairs[i + 1]["price"]
                cur = p["price"]
                diff1 = abs(cur - prev)
                diff2 = abs(cur - _next)
                if (
                    diff1 > correct_group_price_thres
                    and diff2 > correct_group_price_thres
                ):
                    _pairs[i]["price"] = _pairs[i]["orin_price"]

        return _pairs
