from src.utils import (
    is_combo_menu,
    is_price,
    has_combo,
    extract_price,
    clean_price,
    get_price,
)
from src.utils import extract_boxes_same_line_local
from src.utils import Coors, Point, Food

import glob, pathlib, json
from PIL import Image, ImageDraw


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
    # Sort instances by y_min

    results = list(filter(lambda x: x[1].upper() != "HOT", results))

    results = sorted(results, key=lambda x: x[0][1])

    flag_combo_menu = is_combo_menu(results=results)

    food_list = list()
    print("flag_combo_menu:", flag_combo_menu)
    if flag_combo_menu:

        for i in range(len(results)):

            points, text, conf = results[i]
            if is_price(text):
                print(text)

                combo_coors = results[i][0]
                name = results[i][1]
                flag_combo = has_combo(name)
                if flag_combo:
                    price = extract_price(text)
                    price = clean_price(price)
                    combo_price = get_price(text)
                    price_coors = Coors(*[Point(x, y) for (x, y) in combo_coors])

        for i in range(len(results)):

            coors = results[i][0]
            name = results[i][1]
            flag_combo = has_combo(name)
            if flag_combo:
                coors = Coors(*[Point(x, y) for (x, y) in coors])
                food = Food(
                    coors=coors,
                    price_coors=price_coors,
                    name=name,
                    price=combo_price,
                    combo=flag_combo,
                )
                food_list.append(food)

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

            line_boxes, checked = extract_boxes_same_line_local(a, b, results, checked)

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

                    price_coors = Coors(*[Point(x, y) for (x, y) in curr_price[0]])
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


if __name__ == "__main__":

    image_list = glob.glob("test_images/*.jpeg")
    image_list = sorted(image_list)
    # print(image_list)
    for image_path in image_list:
        print(image_path)
        image = Image.open(image_path).convert("RGB")

        image_drawer = ImageDraw.Draw(image)

        ocr_path = pathlib.Path("output") / (
            pathlib.Path(image_path).stem + ".jpeg.json"
        )

        ocr_results = json.load(open(ocr_path))
        pairs = extract_pairs(ocr_results)

        for food in pairs:
            # if food.combo:
            food_center = food.coors.center()
            price_center = food.price_coors.center()
            image_drawer.line([food_center, price_center], fill="red")
            # input()

        # print(f"out/output_{pathlib.Path(image_path).stem}.jpeg")
        image.save(f"out/output_{pathlib.Path(image_path).stem}.png")

        # input()
