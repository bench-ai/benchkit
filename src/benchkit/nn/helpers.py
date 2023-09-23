import os


def create_model_dir():
    if os.path.isdir("./Models"):
        pass
    else:
        current_path = "./Models"
        os.mkdir(current_path)

        whole_path = os.path.join(current_path, "ProjectModels.py")
        init_path = os.path.join(current_path, "__init__.py")

        with open(init_path, "w"):
            pass

        with open(whole_path, "w") as file:
            file.write("import torch\n")
            file.write("import torch.nn as nn\n")
            file.write("\n")
            file.write("\n")
            file.write("# Write your models here")
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("\n")
