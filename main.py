import json
import base64
from src.extractor import Extractor
from flask import Flask, request, json
from src.utils import parse_yaml
import cv2
import numpy as np
from traceback import format_exc

app = Flask(__name__)

cfg = parse_yaml("cfg/app.yaml")

extractor = Extractor(
    detector_weights=cfg.detector_weights,
    recognizer_weights=cfg.recognizer_weights,
    vietocr_cfg=cfg.vietocr_cfg,
    batch_size=cfg.batch_size,
    pt=cfg.pt,
)


@app.route("/healthCheck", methods=["GET"])
def health_check():
    """
    Health check the server
    Return:
    Status of the server
        "OK"
    """
    return "OK"


@app.route("/infer", methods=["POST"])
def infer():
    """
    Do inference on input image
    Return:
    Dictionary Object following this schema
        {
            "image_name": <Image Name>
            "infers":
            [
                {
                    "food_name_en": <Food Name in Englist>
                    "food_name_vi": <Food Name in Vietnamese>
                    "food_price": <Price of food>
                }
            ]
        }
    """

    # Read data from request
    image_name = request.form.get("image_name")
    encoded_img = request.form.get("image")

    # Convert base64 back to bytes
    img = base64.b64decode(encoded_img)

    # Convert bytes to np.array
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        pairs = extractor.extract_menu(img)
        response = {"image_name": image_name, "infers": []}
        for pair in pairs:
            dct = {
                "food_name_en": pair["en_name"],
                "food_name_vi": pair["vi_name"],
                "food_price": pair["price"],
            }
            response["infers"].append(dct)
        return json.dumps(response)

    except Exception as err:
        print(str(err))
        print(format_exc())
        return None


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
