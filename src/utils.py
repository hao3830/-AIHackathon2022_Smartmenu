from typing import Union, List
import re
import yaml
import yaml
from easydict import EasyDict
from unidecode import unidecode
from pathlib import Path

import dataclasses

import numpy as np

# from PIL import Image, ImageDraw
from shapely.geometry import Polygon

PRICE_REGEX = "\d{1,6}([,.]?\d{3}[dđ]?|k?)?|mien phi|free"


def parse_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return EasyDict(data)


def is_price(text):
    text = text.lower()
    text = unidecode(text)
    result = re.search(PRICE_REGEX, text)
    if result:
        price = result.group()
        price_idx = text.index(price)
        if price_idx == 0:
            return True

        if text[price_idx - 1] != " ":
            return False

        return True

    return False


def extract_price(text) -> Union[str, None]:
    text = text.lower()
    text = unidecode(text)
    result = re.search(PRICE_REGEX, text)
    if result:
        return result.group()
    return None


def get_price(text):
    price = extract_price(text)
    price = clean_price(price)

    return price


def extract_boxes_same_line(a, b, results, checked):
    boxes = []
    for i in range(len(results)):
        if i in checked:
            continue

        points, text, conf = results[i]

        # check if any box with xmin, ymin that the line cross through
        if (
            points[0][0] * a + b >= points[0][1]
            and points[3][0] * a + b <= points[3][1]
        ):
            checked[i] = True
            boxes.append(results[i])

    if len(boxes) <= 1:
        return None, checked

    return boxes, checked


# def extract_pairs(results):
#     """extract pairs of food name and price

#     Args:
#         results: points[xy, xy, xy, xy], text, det_conf

#     Returns:
#         _type_: _description_
#     """
#     pair_text_price = []
#     index = 0
#     checked = {}
#     checked["index"] = 0
#     # Sort instances by y_min

#     results = list(filter(lambda x: x[1].upper() != "HOT", results))

#     results = sorted(results, key=lambda x: x[0][1])
#     for i in range(len(results)):

#         points, text, conf = results[i]
#         if i in checked:
#             continue

#         if not is_price(text):
#             continue

#         y_center_left = (points[0][1] + points[3][1]) / 2
#         y_center_right = (points[1][1] + points[2][1]) / 2
#         x_left = points[0][0]
#         x_right = points[2][0]

#         # Straight line formular: y = ax + b

#         a = (y_center_right - y_center_left) / (x_right - x_left)
#         b = y_center_left - a * x_left

#         line_boxes, checked = extract_boxes_same_line(a, b, results, checked)

#         if not line_boxes:
#             continue

#         curr_name = None
#         curr_price = None
#         visited = [0] * len(line_boxes)

#         for i in range(len(line_boxes)):
#             if visited[i]:
#                 continue
#             visited[i] = 1

#             checked[i] = True
#             if is_price(line_boxes[i][1]):
#                 curr_price = line_boxes[i]
#             else:
#                 curr_name = line_boxes[i]

#             if not curr_name:
#                 for j in range(i + 1, len(line_boxes)):
#                     if visited[j] or is_price(line_boxes[j][1]):
#                         continue
#                     else:
#                         curr_name = line_boxes[j]
#                         visited[j] = 1

#             if curr_name and curr_price:
#                 pair_text_price.append([curr_name, curr_price])
#                 index += 1
#                 curr_name = None
#                 curr_price = None

#     # Sort by y_min for better view
#     pair_text_price = sorted(pair_text_price, key=lambda x: x[0][0][0][1])

#     return pair_text_price


def clean_price(price):
    price = price.lower()

    price = extract_price(price)

    price = price.replace(" ", "")
    price = price.replace(",", "")
    price = price.replace(".", "")
    price = price.replace("đ", "")
    price = price.replace("d", "")
    if price.endswith("k"):
        price = price.replace("k", "")
        price += "000"

    price = unidecode(price)
    if "mien phi" in price or "free" in price:
        return 0

    if len(price) < 4:
        price += "000"

    return int(price)


def load_reference_csv(csv_path, translate=False):
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    referenced_translate = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            # ImageName,VietnameseName,EnglishName,Price
            line = line.replace(" ,", ",")
            line = line.strip().split(",")
            # Advanced split to ignore comma in quote
            # line = [s.strip(",")
            #         for s in line.split('"') if s and s != ","]
            if len(line) != 4:
                continue
            if line[0] == "ImageName":  # Skip the header line
                continue

            k = line[1].strip().lower()
            if translate:
                v = line[2].strip().lower()
                v = v.strip('"')
            else:
                v = [i.strip('"') for i in line]
                # Skip price is 15000 - 20000
                if "-" in v[-1]:
                    continue

            if k not in referenced_translate:
                referenced_translate[k] = v
            elif len(referenced_translate[k]) < len(k):
                i = 0
                while True:
                    _t = f"{k}_{i}"
                    if _t in referenced_translate:
                        i += 1
                        continue
                    referenced_translate[_t] = v
                    break

    return referenced_translate


_COMBO_FLAGS = ["COMBO", "ĐỒNG GIÁ"]


def get_box_center(bboxes: List):
    """
    Parameter
    ---------
    bboxes: Array of coordinates of shape 4 x 2
    Return
    ------
    xy: Tuple of center coordinates of bboxes
    """
    bboxes = np.array(bboxes)
    assert bboxes.shape == (4, 2)

    center = list(Polygon(bboxes).centroid.coords)[0]
    center = tuple(map(int, center))
    return center


@dataclasses.dataclass
class Point:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y


@dataclasses.dataclass
class Coors:
    p_1: Point
    p_2: Point
    p_3: Point
    p_4: Point

    def __iter__(self):
        self.l = [self.p_1, self.p_2, self.p_3, self.p_4]

        self.l = [tuple(point) for point in self.l]
        i = 0
        while i < 4:
            yield self.l[i]
            i += 1

    def center(self):
        _coors = list(self)
        _center = get_box_center(_coors)
        return _center


@dataclasses.dataclass
class Food:
    coors: Coors
    price_coors: Coors
    name: str
    price: int
    combo: bool

    def center(self):
        _center = list(Polygon(self.coors).centroid.coords)[0]
        return _center


def has_combo(text: str):
    for flag in _COMBO_FLAGS:
        result = re.search(flag.lower(), text)
        if result:
            return True
    return False


def is_combo_menu(results: List) -> bool:
    """
    Parameter
    ---------
    bboxes: Array of coordinates of shape 4 x 2
    Return
    ------
    True if it a menu with combo price
    False otherwise
    """

    food_list = list()
    price_list = list()
    combo_list = list()
    for i in range(len(results)):
        points, text, conf = results[i]

        if_combo = has_combo(text)
        if if_combo:
            if_price = is_price(text)
            if not if_price:
                continue
            combo_list.append(text)
            continue

        if_price = is_price(text)
        if if_price:
            price_list.append(text)
            continue

        food_list.append(text)

    if len(combo_list) == 0:
        return False
    return True


def extract_pairs(results):
    """extract pairs of food name and price

    Args:
        results: points[xy, xy, xy, xy], text, det_conf

    Returns:
        _type_: _description_
    """
    pair_text_price = []
    index = 0
    checked = {}
    checked["index"] = 0

    results = list(filter(lambda x: x[1].upper() != "HOT", results))

    # Sort instances by y_min
    results = sorted(results, key=lambda x: x[0][0][1])

    flag_combo_menu = is_combo_menu(results=results)

    food_list = list()
    if flag_combo_menu:

        # get combo price for all combo below until reach new combo price
        combo_price = None
        price_coors = None
        for i in range(len(results)):
            points, text, conf = results[i]

            _is_price = is_price(text)
            _is_combo = has_combo(text)

            if _is_price and _is_combo:
                combo_price = get_price(text)
                price_coors = Coors(*[Point(x, y) for (x, y) in points])
            elif _is_combo and combo_price is not None:
                # Combo below combo price tag
                coors = Coors(*[Point(x, y) for (x, y) in points])
                food = Food(
                    coors=coors,
                    price_coors=price_coors,
                    name=text,
                    price=combo_price,
                    combo=_is_combo,
                )
                food_list.append(food)

        # for i in range(len(results)):
        #     coors = results[i][0]
        #     name = results[i][1]
        #     flag_combo = has_combo(name)
        #     if flag_combo:
        #         coors = Coors(*[Point(x, y) for (x, y) in coors])
        #         food = Food(
        #             coors=coors,
        #             price_coors=price_coors,
        #             name=name,
        #             price=combo_price,
        #             combo=flag_combo,
        #         )
        #         food_list.append(food)

    else:

        for i in range(len(results)):

            points, text, conf = results[i]
            if i in checked:
                continue

            if not is_price(text):
                continue

            y_center_left = (points[0][1] + points[3][1]) / 2
            y_center_right = (points[1][1] + points[2][1]) / 2
            x_left = points[0][0]
            x_right = points[2][0]

            # Straight line formula: y = ax + b

            a = (y_center_right - y_center_left) / (x_right - x_left)
            b = y_center_left - a * x_left

            line_boxes, checked = extract_boxes_same_line(
                a, b, results, checked)

            if not line_boxes:
                continue

            curr_name = None
            curr_price = None
            visited = [0] * len(line_boxes)

            for i in range(len(line_boxes)):
                if visited[i]:
                    continue
                visited[i] = 1

                checked[i] = True
                if is_price(line_boxes[i][1]):
                    curr_price = line_boxes[i]
                else:
                    curr_name = line_boxes[i]

                if not curr_name:
                    for j in range(i + 1, len(line_boxes)):
                        if visited[j] or is_price(line_boxes[j][1]):
                            continue
                        else:
                            curr_name = line_boxes[j]
                            visited[j] = 1

                if curr_name and curr_price:
                    pair_text_price.append([curr_name, curr_price])

                    name = curr_name[1]
                    flag_combo = has_combo(name)
                    coors = Coors(*[Point(x, y) for (x, y) in curr_name[0]])

                    price_coors = Coors(*[Point(x, y)
                                        for (x, y) in curr_price[0]])
                    price = get_price(text)
                    food = Food(
                        coors=coors,
                        price_coors=price_coors,
                        name=name,
                        price=price,
                        combo=flag_combo,
                    )
                    food_list.append(food)

                    index += 1
                    curr_name = None
                    curr_price = None

    return food_list


def load_vietocr_config_from_file(fname):
    with open("./cfg/base.yaml") as f:
        base_config = yaml.safe_load(f)
    with open(fname, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    base_config.update(config)

    return dict(base_config)
