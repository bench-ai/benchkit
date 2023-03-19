import json
from PIL import Image
import os
import pathlib


def iterate_directory(file_dir: str):
    with os.scandir(file_dir) as walk:
        for i in walk:
            if os.path.isfile(i):
                yield i.name
            elif os.path.isdir(i):
                new_path = pathlib.Path(file_dir).resolve() / i.name
                iterate_directory(new_path)



def save_inference(data_input: list[str],
                   result: list[str]) -> None:
    def check_file_validity(file_path: str):

        img_endings = ("png", "jpeg", "jpg")

        try:
            with open(file_path, "r") as file:
                if file_path.endswith("json"):
                    try:
                        json.load(file)
                    except json.JSONDecodeError:
                        raise ValueError("Json file improperly is not valid")
                elif file_path.lower().endswith(img_endings):
                    try:
                        Image.open(file_path)
                    except IOError:
                        raise ValueError("Invalid Image File present")

        except FileNotFoundError:
            raise ValueError("File does not exist")

    # operations that take place after a model has parsed the data
    # Saves the data and displays it
    # Can only save two kinds of results, images and Json

    for i in data_input + result:

        try:
            check_file_validity(i)
        except ValueError:
            # Log the inability to save the inference
            pass

    # this should save the data in the appropriate epoch folder
    save_inference_s3(data_input, result)


# tomorrow setup logging
# setup one epoch data checking
# setup method that lets users upload their inference data(limit the amount)
# setup the data upload process
# make cat and dog dataset

if __name__ == '__main__':
    for idx, i in enumerate(iterate_directory("/Users/sriramgovindan/Documents/Bench-datasets/dogs-vs-cats")):
        if idx < 3:
            print(i)

