import os


def create_model_dir():
    if os.path.isdir("./models"):
        pass
    else:
        current_path = "./models"
        os.mkdir(current_path)

        whole_path = os.path.join(current_path, "project_models.py")
        init_path = os.path.join(current_path, "__init__.py")

        with open(init_path, "w"):
            pass

        with open(whole_path, "w") as file:
            file.write("# Write your models here")
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("\n")
