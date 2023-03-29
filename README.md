# *TorchBench*
Welcome to TorchBench the Bench SDK for PyTorch - a cloud integrated 
library that enables you to leverage your PyTorch deep learning models on the 
cloud through a PyTorch-based API.
## Table of Contents
* [Installation](#installation)
  * [Requirements](#requirements)
  * [Build from Source](#build-from-source)
  * [Pip Install](#pip-install)
  * [Check install](#check-install)
* [Authentication](#authentication)
  * [Login](#login)
  * [Logout](#logout)
* Register a Project on [Bench](https://bench-ai.com/)
* Define a Model
* Dataloaders
* Logging(Coming Soon)
* Graphing(Coming Soon)
* Optimization(Coming Soon)
* Migrate your project
    * Migrating your settings
	* Spawning your dedicated server
    * Transferring your data to a bucket / using a hosted dataset
	*  Transferring your project data to your Bench Project
* Deploy your trained model for inference(Coming Soon)
___


## Installation
#### Requirements
- Requires [python 3.10](https://www.python.org/)
- Requires [Docker](https://www.docker.com/)
#### Build from Source
Create a virtual environment
```bash
# Create the virtual environment
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate

# install all requirements
pip install -U pip setuptools wheel
pip install -r requirements.txt
```
#### Pip install
#### Check Install
```bash
python MainCLI.py --version
```
___

## Authentication
We provide two ways to configure your settings the more persistent
method is to specify your credentials in a Config.json file.
This config file should be in your root directory, and the
data in it will be confidential, so take caution.

Logging in is required to use any core functionalities in the
library

### Login
Config(recommended)
```json
{
	"user_credentials": {
		"username": "username", 
		"password": "password"
	}
}
```
```bash
python MainCLI.py --login
```

Manual
```bash
python MainCLI.py --loginm
```

### Logout
Manual
```bash
python MainCLI.py --loginm
```
___