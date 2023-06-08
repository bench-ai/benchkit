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

To connect your Bench project to your codebase simply provide your project id

### Start Project
All code that you wish to migrate to the cloud must be written
inside a project.

Projects have three key features
- Models
- Dataloaders
- Training Script

All Key Parts familiar to any ML practitioner

```bash
bench-kit startproject <project_id> <api_key>
```
If you forget your apikey you can generate a new one, in your project settings


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

In regular pytorch you would make a Dataset like this

```python
class CatDog(Dataset):
    
    def __init__(self,
                 dataset_path: str,
                 data_list: list[str],
                 transforms):
        
        self._dataset_path = dataset_path
        self._data_list = data_list
        self._transforms = transforms

    @staticmethod
    def convert_label_to_numeric(animal: str) -> int:
        return 0 if animal == "cat" else 1

    def __len__(self):
        return len(self._data_list)

    def get_label_and_numeric_data(self, item):
        split = self._data_list[item].split(".")[0]
        assert len(split) == 3

        return self.convert_label_to_numeric(split)

    def __getitem__(self, idx: int):
        file: str = self._dataset_path[idx]
        label = self.get_label_and_numeric_data(file)
        file = os.path.join(self._dataset_path, file)
        
        return label, self._transforms(file)
        
```

In this class the user provides a list of file names i.e
["cat_one.png", "dog_two.png"], the base path to the directory with the files
and a transformations they want to conduct on those files.

Then based on the length of the list of files pytorch will generate a number between
[0, len(the_list)] and will use that number to randomly pick a file to serve to your model.

In the getitem method we first use the name of the file to get the label it belongs too(in this case a cat or dog).
We then return the label along with the image applied with the appropriate transformations.

Using bench-kit this process largely stays the same but some changes need to be made to make your data cloud ready.

1) We need an elegant system to transfer the data to the cloud efficiently 
2) We need a method to get the data from the cloud in a cost-efficient manner.

To achieve this we have to split up our Singular dataset in the first example
into 2 easy to use classes.

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

By implementing this class you allow us to split your many files into large chunks for upload.
Saving you countless dollars on data access costs, and countless hours in data transport.

Now that you have your code ready for cloud transport we need a way to unpack it for access when training your model.
That's where the next dataset comes into play.

#### IterableChunk

This is the dataset your model will use. It works
by unpacking your data that has been transported to the cloud.

To make a IterableChunk dataset you need to inherit the ChunkDataset
class, and you need to override one method.

##### _data_iterator

Data Iterator is the iterator that returns samples from your data. Since
this is the exact data that will be fed into your model, any required transformation or augmentation should
be applied.

In data iterator you need to make a super call to **_data_iterator** this will return you 
the data you specified in ProcessorDataset, I.E. files and labels or just
labels.

#### Example

Here is an example of how the IterableChunk dataset would look like, following
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

From these two examples you can see that all we did was split up one dataset class
into two parts one for transporting your data based on its location and content, and one for serving it to your model.

Once you have your classes ready you will need to declare them.

In the main method return a list

In this list pass in 
- The ProcessorDataset **object**
- The ChunkDataset **class**
- Dataset name
- args for ChunkDataset
- kwargs for ChunkDataset

Do this for every dataset you wish to migrate to the cloud.

#### Example
```python
def main():
    # Write your data loading code here
    data = [
        (CatDog(d_path, train_list), CatDogChunk, "train", [True], {}),
        (CatDog(d_path, val_list), CatDogChunk, "val", [False], {})
    ]

    return data
```
Finally, to migrate your data to the cloud run 
```bash
python manage.py migrate-data
```

This will migrate all datasets in the list to migrate only one dataset run
```bash
python manage.py migrate-data <ds name>
```

Migration is split into three parts each of which are handled entirely for you
1) Chunking your dataset
2) Testing the IterableChunk with the chunked dataset
3) Uploading the dataset

By default, when running the above command these processes are done consecutively,
However if you wish to run each command independently this can be achieved.

##### Chunk

To chunk your dataset run

```bash
python manage.py migrate-data --zip
```

##### Test

To test your dataset run

```bash
python manage.py migrate-data --tdl
```

##### Upload

To Upload your dataset run

```bash
python manage.py migrate-data --up
```

##### General
To see all datasets you have Uploaded run
```bash
bench-kit show-ds
```

To delete a dataset you have Uploaded run
```bash
bench-kit del-ds
```

## TrainScript

To facilitate multi-gpu training changes have to be made to your code.
To achieve this libraries are often incorporated, the library we use is [Hugging Face
Accelerate](https://huggingface.co/docs/accelerate/index)

A more intricate explanation on how to use the library can be found on their site, but we will give you a basic breakdown as well.

First you need to get a dataset to achieve this, to do this call the get dataset method
```python
from BenchKit.Data.Helpers import get_dataset

train_dataset: DataLoader = get_dataset(CatDogChunk,
                                            True,
                                            "train",
                                            tracker_config["bs"],
                                            2,
                                            True)


val_dataset: DataLoader = get_dataset(CatDogChunk,
                                      True,
                                      "val",
                                      tracker_config["bs"],
                                      2,
                                      False)
```

Then just like regular pytorch declare your model, loss_fn, optim. To 
get these pieces of data ready for distributed training, you have to use
the accelerate prepare method.

```python
from accelerate import Accelerator
from BenchKit.Train.Helpers import get_accelerator

model = CDResNet(64)
loss_fn = nn.BCELoss()
optim = opt.Adam(params=model.parameters(), lr=tracker_config["lr"])

acc: Accelerator = get_accelerator()
model, loss_fn, optim, train_dataset, val_dataset = acc.prepare(model,
                                                                loss_fn,
                                                                optim,
                                                                train_dataset,
                                                                val_dataset)
```

From here train and validate your model
```python
for _ in range(tracker_config["epochs"]):
    train_one_epoch(acc, train_dataset, model, optim, loss_fn)
        
    validate_one_epoch(acc, val_dataset, model, val_length)
```

Here is the train method, the main things that need to be changed is that instead of using a torch device object
one would use an accelerator.device object. You would also want to call the wipe_temp method
to remove any lingering files. If you are using a multi gpu setup, it is recommended to run
accelerate.gather this method will combine all the losses from all gpu's for one efficient calculation.
```python
def train_one_epoch(accelerate: Accelerator,
                    train_dl: DataLoader,
                    model,
                    optim,
                    loss_fn):
    
    model.train()
    total_loss = 0
    count = 0
    
    for batch in train_dl:
        
        optim.zero_grad()
        
        targets, inputs = batch
        outputs = model(inputs)
        
        targets = targets.type(torch.FloatTensor)
        targets = targets.to(accelerate.device)

        loss = loss_fn(outputs, targets)
        accelerate.backward(loss)
        optim.step()
        
        all_loss, outputs = accelerate.gather((loss, outputs))
        total_loss += torch.reshape(all_loss, (-1,)).item()
        count += torch.tensor(outputs.size()[0], device=accelerate.device)

    print(total_loss / count)
    wipe_temp(accelerate)
```

Same process for the validation method
```python
def validate_one_epoch(accelerate: Accelerator,
                       test_dl: DataLoader,
                       model,
                       length):
    model.eval()
    total_correct = 0
    total_length = 0
    for batch in tqdm(test_dl,
                      colour="blue",
                      total=length + 1,
                      disable=not accelerate.is_local_main_process):

        accelerate.free_memory()
        targets, inputs = batch
        outputs = model(inputs)
        targets = targets.type(torch.FloatTensor)
        targets = targets.to(accelerate.device)

        class_count, length = get_class_loss(accelerate, outputs, targets)
        class_count, length = accelerate.gather((class_count, length))

        total_length += torch.sum(torch.reshape(length, (-1,)))
        total_correct += torch.sum(torch.reshape(class_count, (-1,)))

    val_loss: torch.Tensor = total_correct / total_length
    wipe_temp(accelerate)
```

### Simple messaging

To send logs to your Experiment dashboard, you will need to instantiate a tracker with your accelerator
```python
tracker_config = {
        "epochs": 10,
        "bs": 16,
        "lr": 1e-2
    }


tracker = BenchAccelerateTracker("trial tun",tracker_config["epochs"])
acc: Accelerator = get_accelerator(log_with=tracker)
acc.init_trackers("my_project", config=tracker_config)
```
The tracker config can contain any data you wish, but we recommend using hyperparameters as the config
since the config will be the first sent message.

To log a message with the tracker use this method
```python
accelerate.log()
```

### Checkpointing
To save checkpoints of your model use this method
```python
upload_model_checkpoint()
```

To see all your checkpoints run this command
```bash
bench-kit show-check
```

To get a specific checkpoint run this command
```bash
bench-kit get-check
```

To delete a specific checkpoint run this command
```bash
bench-kit del-check
```

A full example of the TrainScript can be seen [here]("https://github.com/Bench-ai/CatDogBenchKit/blob/main/TrainScript.py")

## Cloud Migration

Congrats you are now ready to migrate your bench project, for some
_fast_ cloud training.

This can all be done with one simple line
```bash
python manage.py migrate-code
```

This line will package your code ready for the cloud however we also support 
versioning so if you want to pass in a specific version of your code. Use this line.

```bash
python manage.py migrate-code <version: int>
```

If you wish to see all uploaded versions type
```bash
bench-kit show-vs
```

If you wish to delete a specific version of your code run
```bash
bench-kit del-vs
```


