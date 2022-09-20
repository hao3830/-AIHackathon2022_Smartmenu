import cv2
import time

from pathlib import Path
from utils import line_process
from text_reader import Reader
from translator import Translator
import easyocr


def main():
    source = Path("../yolov5/dataset/aihackathon_alltext/val_images")
    yolo_weights = "../yolov5/runs/train/yolov5l_alltext_200ep/weights/best.onnx"
    vietocr_weights = Path("vietocr_weights")
    save_dir = Path("visualize")

    visualize = True

    if visualize:
        save_dir.mkdir(parents=True, exist_ok=True)

    reader = Reader(
        detector_weights=yolo_weights,
        recognizer_weights=vietocr_weights,
    )
    # reader2 = easyocr.Reader(['vi', 'en'], gpu=False)

    translator = Translator(
        referenced_csv="./label.csv",
    )

    total_time = 0.0
    for img_idx, img_path in enumerate(sorted(source.iterdir())):
        tic = time.time()
        img0 = cv2.imread(str(img_path))

        boxes = reader(img0)
        # boxes = list(reader2.readtext(img0))

        mapping_text_price = line_process(boxes)

        if visualize:
            for box, text, conf in boxes:
                pt1 = tuple(map(round, box[0]))
                pt2 = tuple(map(round, box[2]))
                res = cv2.rectangle(img0, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

                # Testing purpose only,
                # just translate food name, not all detected text
                # translated_text = translator(text)
                # print(text, translated_text)
            for key in mapping_text_price:
                cv2.line(
                    res,
                    tuple(mapping_text_price[key][0][0][0]),
                    tuple(mapping_text_price[key][1][0][0]),
                    (0, 0, 255),
                    2,
                )

        if visualize:
            cv2.imwrite(str(save_dir / img_path.name), res)

        toc = time.time()
        process_time = toc - tic
        total_time += process_time
        print(f"Infer time: {process_time:.3f}s")

        if img_idx == 1:
            break

    print(total_time / 1)


if __name__ == "__main__":
    main()
