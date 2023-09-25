import os


def create_model_dir():
    if os.path.isdir("models"):
        pass
    else:
        os.mkdir("models")

        whole_path = os.path.join("models", "project_models.py")
        init_path = os.path.join("models", "__init__.py")

        with open(init_path, "w"):
            pass

        with open(whole_path, "w") as file:
            file.write("# Write your models here")
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("\n")
