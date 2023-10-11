import os
import unittest

from src.benchkit.data.helpers import create_dataset_dir


class TestCreateDatasetDir(unittest.TestCase):
    def test_create_dataset_dir(self):
        # Run the function
        create_dataset_dir()

        # Check if the 'datasets' directory is created
        self.assertTrue(os.path.isdir("datasets"))

        # Check if the '__init__.py' file is created
        init_path = os.path.join("datasets", "__init__.py")
        self.assertTrue(os.path.isfile(init_path))

        # Check if the 'project_datasets.py' file is created
        whole_path = os.path.join("datasets", "project_datasets.py")
        self.assertTrue(os.path.isfile(whole_path))


if __name__ == "__main__":
    unittest.main()
