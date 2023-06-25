FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

# Update package lists
RUN apt-get update

# Install dependencies
RUN apt-get install -y software-properties-common wget
# Add the deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa
# Update package lists again
RUN apt-get update
# Install Python 3.10 and pip
RUN apt-get install -y python3.10 python3.10-distutils
# Set Python 3.10 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

RUN apt-get update
RUN apt-get install -y git

COPY ./Datasets /Bench-Project/Datasets
COPY ./Models /Bench-Project/Models

COPY ./TrainScript.py /Bench-Project/
COPY ./manage.py /Bench-Project/
COPY ./dependencies.txt /Bench-Project/
COPY ./entrypoint.sh /Bench-Project/

WORKDIR /Bench-Project

RUN chmod +x entrypoint.sh
CMD ["/Bench-Project/entrypoint.sh"]









