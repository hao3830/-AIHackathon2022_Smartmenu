import requests
import base64
import os
import json
import sys
import base64
import time
from pathlib import Path
from traceback import format_exc


def measure_time(f):
    def inner_func(*args, **kargs):
        tic = time.time()
        res = f(*args, **kargs)
        toc = time.time()
        print(f"Processed time: {toc - tic:.3f}s")
        return res

    return inner_func


@measure_time
def send_request(img_path):
    url = "http://localhost:5002/infer"
    with img_path.open("rb") as f:
        img = base64.b64encode(f.read())

    try:
        response = requests.post(
            url,
            data={
                "image": img,
                "image_name": img_path.name,
            },
        )
        return response
    except Exception as err:
        print(str(err))
        print(format_exc())
        return


def test_multiple_images():
    imgs_dir = Path("./test_images")
    save_dir = Path("./output")
    save_dir.mkdir(parents=True, exist_ok=True)

    while True:
        action = input("Img?: ")
        if action == "q":
            exit()

        for idx, img_path in enumerate(sorted(imgs_dir.iterdir())):
            if action not in img_path.name:
                continue
            print(f"Processing image {img_path.name}")
            res = send_request(img_path)
            if res is None or res.status_code != 200:
                print(f"Request failed for image {img_path.name}")
                print("Response content")
                print(res)
                continue

            save_path = save_dir / (img_path.stem + ".json")
            json.dump(
                res.json(),
                save_path.open("w", encoding="utf-8"),
                indent=4,
                ensure_ascii=False,
            )


def main():
    tic = time.time()
    for i in range(1):
        test_multiple_images()
    toc = time.time()
    total_time = toc - tic
    print(f"Total time: {total_time}s")
    print(f"Avg time: {total_time / 100}")


if __name__ == "__main__":
    main()
