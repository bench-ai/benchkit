# *Bench-Kit*
Welcome to Bench-Kit! The Bench SDK for PyTorch - a cloud integrated 
library that enables you to leverage your PyTorch deep learning models on the 
cloud through a PyTorch-Django based framework.
## Table of Contents
* [Installation](#installation)
  * [Requirements](#requirements)
  * [Build from Source](#build-from-source-manual-build)
  * [Pip Install Nightly Build](#install-nightly-build-recommended-build)
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
* Model Tracking(Coming Soon)
* Optimization(Coming Soon)
* [Migrate your project](#cloud-migration)
* Deploy your trained model for inference(Coming Soon)
___


## Installation
#### Requirements
- Requires [python 3.10](https://www.python.org/)
- Requires [Docker](https://www.docker.com/)
- Requires Docker to be accessed without sudo as seen [here](https://docs.docker.com/engine/install/linux-postinstall/)
- Requires [Bench ai account](https://bench-ai.com/signup)

#### Create a virtual environment
```bash
# Create the virtual environment
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate

# update the build tools
pip install -U pip setuptools wheel
```
##### Build from source (MANUAL BUILD)
Clone the repository

Run the following to install dependencies
```bash
pip install -r requirements.txt
```

##### Install Nightly Build (RECOMMENDED BUILD)
```bash
pip install git+https://github.com/Bench-ai/BenchKit.git
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
bench-kit startproject <project_name>
```

### Logout
```bash
bench-kit logout
```
---
## Dataloaders
When starting a project you should see a Datasets folder with a 
ProjectDatasets file.

In this file you have to declare two types of datasets
1) ProcessorDataset
2) IterableChunk

#### ProcessorDataset

This class tells us how to traverse your directory structure,
so we know which files are needed for transport to the cloud,
and how they should be packaged.

To make a Processor Dataset you need to inherit the ProcessorDataset
class, and you need to override three methods

1) If your dataset uses large files to train
such as images. You will need to override the
**get_file** method
   1) This Method has to return 
   a list of strings containing the full path to your file
   2) This method should be implemented if your dataset consists of, image files, wav files, txt files, etc.
2) You will need to also override **get_label_and_numeric_data** This method
packages your labels / small numeric data in a tensor file. 
   1) This means you should convert your labels to the exact tensor values
   you wish to feed into your model.
   2) This would consist of class numbers, bounding box coordinates, or any other numeric data
   that does not warrant its own file
3) You need to also override **__len_\_** this lets us know the size
of your dataset.

#### Example

Here is an example of how the ProcessorDataset dataset would look like
for a Cat Dog object detection dataset

```python
class CatDog(ProcessorDataset):

    def __init__(self,
                 dataset_path: str,
                 data_list: list[str]):
        self._dataset_path = dataset_path
        self._data_list = data_list

    @staticmethod
    def convert_label_to_numeric(animal: str) -> int:
        return 0 if animal == "cat" else 1

    def __len__(self):
        return len(self._data_list)

    def get_label_and_numeric_data(self, item):
        split = self._data_list[item].split(".")[0]
        assert len(split) == 3

        return self.convert_label_to_numeric(split)

    def get_file(self, item) -> list[str]:
        return [os.path.join(self._dataset_path, self._data_list[item])]
```

#### IterableChunk

This is the dataset your model will use. It works
by unpacking your data that has been transported to the cloud.

To make a ChunkDataset you need to inherit the ChunkDataset
class, and you need to override one method.

#### _data_iterator

Data Iterator is the iterator that returns samples from your data. Since
this is the exact data that will be fed into your model, any required transformation or augmentation should
be applied.

In data iterator you need to make a super call to **_data_iterator** this will return you 
the data you specified in ProcessorDataset, I.E. files and labels or just
labels.

#### Example

Here is a example of how the IterableChunk dataset would look like, following
the previous example

```python
class CatDogChunk(IterableChunk):

    def __init__(self,
                 name: str,
                 cloud: bool,
                 is_train: bool):

        super().__init__(name, cloud)

        mean = [0.4766, 0.4527, 0.3926]
        std = [0.2275, 0.2224, 0.2210]

        self._trans = tf.Compose([tf.Resize((256, 256)),
                                  tf.RandomHorizontalFlip(),
                                  tf.ToTensor(),
                                  tf.Normalize(mean, std)]) if is_train else tf.Compose([tf.Resize((256, 256)),
                                                                                         tf.ToTensor(),
                                                                                         tf.Normalize(mean, std)])

    def _data_iterator(self):
        for labels, file in super()._data_iterator():
            img_pil = Image.open(file[0]).convert('RGB')
            yield labels, self._trans(img_pil)
```

#### Putting it all together

Once you have your IterableChunk and your ProcessorDataset, it's
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


