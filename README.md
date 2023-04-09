# *Bench-Kit*
Welcome to Bench-Kit the Bench SDK for PyTorch - a cloud integrated 
library that enables you to leverage your PyTorch deep learning models on the 
cloud through a PyTorch-Django based framework.
## Table of Contents
* [Installation](#installation)
  * [Requirements](#requirements)
  * [Build from Source](#build-from-source)
  * [Pip Install](#pip-install)
  * [Check install](#check-install)
* [Authentication](#authentication)
  * Register a Project on [Bench](https://bench-ai.com/)
  * [Logout](#logout)
* Define a Model
* [Dataloaders](#dataloaders)
  * [ProcessorDataset](#processordataset)
  * [ChunkDataset](#chunkdataset)
  * [Putting it all together](#putting-it-all-together)
* Logging(Coming Soon)
* Graphing(Coming Soon)
* Optimization(Coming Soon)
* [Migrate your project](#cloud-migration)
* Deploy your trained model for inference(Coming Soon)
___


## Installation
#### Requirements
- Requires [python 3.10](https://www.python.org/)
- Requires [Docker](https://www.docker.com/)
- Requires [Bench-ai account](https://bench-ai.com/signup)

#### Create a virtual environment
```bash
# Create the virtual environment
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate

# Upgrade the repo
pip install -U pip setuptools wheel
```
##### Build from source 
```bash
# Clone the repo 
pip install -r requirements.txt
```

##### Pip install
```bash
pip install git+https://github.com/Bench-ai/TorchBench.git
```
#### Check Install
```bash
bench-kit --version
```
___

## Authentication
Logging in is required to use any core functionalities in the
library

To log in simply provide your username and password to your bench 
account upon starting a project

### Start Project
All code that you wish to migrate to the cloud must be written
inside a project.

Projects have three key features
- Models
- Dataloaders
- Training Script

All Key Parts familiar to any ML practitioner

```bash
bench-kit --startproject <project_name>
```

### Logout
```bash
bench-kit --logout
```
---
## Dataloaders
When starting a project you should see a Datasets folder with a 
ProjectDatasets file.

In this file you have to declare two types of datasets
1) ProcessorDataset
2) ChunkDataset

#### ProcessorDataset

This dataset tells us how to traverse your directory structure,
so we can zip your files for easy access

To make a Processor Dataset you need to inherit the ProcessorDataset
class, and you need to override three methods

1) If your dataset works on external files of any sort
such as images. You will need to override the
**get_file** method
   1) This Method has to return 
   a list of strings containing the full path to your file
2) You will need to also override **get_label_and_numeric_data** This method
packages your labels / any numeric data that is not stored in a file, as ready yo use tensor. 
   1) This means you should convert your labels to the exact tensor values
   you wish to feed into your model
2)  You need to also override **__len_\_** this lets us know the size
of your dataset.

#### Example

Here is a example of how the ProcessorDataset dataset would look like
for a Cat Dog object detection dataset

```python
class CatDog(ProcessorDataset):

    def __init__(self):
        self._dataset_path = "path/to/my/ds"
        self._data_list = os.listdir(self._dataset_path)

    @staticmethod
    def convert_label_to_numeric(animal: str) -> int:
        return 0 if animal == "cat" else 1

    def __len__(self):
        return len(self._data_list)

    def get_label_and_numeric_data(self, item):
        split = self._data_list[item].split(".")[0]
        assert len(split) == 3

        return self.convert_label_to_numeric(split[0])

    def get_file(self, item) -> list[str]:
        return [os.path.join(self._dataset_path, self._data_list[item])]

```

#### ChunkDataset

This is the dataset your model will use to train the model, it works
on top of the data provided by the ProcessorDataset.

To make a ChunkDataset you need to inherit the ChunkDataset
class, and you need to override one method.

#### __getitem__

In get item you need to make a super call to **__getitem_\_** this will return you 
the data you specified in ProcessorDataset, I.E. files and labels or just
labels.

From here on out preform the regular transformations you would make in 
any other dataset.

#### Example

Here is a example of how the ChunkDataset dataset would look like
for a Cat Dog object detection dataset

```python
class CatDogChunk(ChunkDataset):

    def __init__(self,
                 name: str):
        super().__init__(name)
        self._trans = tf.Compose([tf.PILToTensor(),
                                  tf.Resize((50, 50), antialias=False)])

    def __getitem__(self, item):
        labels, file = super().__getitem__(item)
        img_pil = Image.open(file[0]).convert('RGB')
        return labels, self._trans(img_pil)
```

#### Putting it all together

Once you have your ChunkDataset and your ProcessorDataset, it's
time to add the last touches.

In the main method make a call to **process_datasets()**

In this method pass in 
- The ProcessorDataset **object**
- The ChunkDataset **class**
- args for ChunkDataset
- kwargs for ChunkDataset

#### Example
```python
def main():
    # Write your data loading code here
    process_datasets(CatDog(), CatDogChunk, "train")
```
---
## Cloud Migration

Congrats you are now ready to migrate your bench project, for some
_fast_ cloud training.

This can all be done with one simple line
```bash
python manage.py --makemigrations
```


