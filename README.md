# Loan Default Detection Model

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Environment Setup](#environment-setup)
- [Docker Instructions](#docker-instructions)


## Project Overview

This repository contains a loan default detection model trained on the data from the Kaggle competition "Home Credit Default Risk". The model aims to predict the likelihood of a loan applicant defaulting on their payments based on various features and historical data.

### Dataset

The dataset used in this project can be obtained from the following Kaggle competition link:
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Please download the dataset from the provided link and place it in the `data` directory before running the model.

### Model Development

The loan default detection model leverages machine learning models and utilizes various features such as applicant information, credit history, and previous loan details to make predictions. The model has been trained using XXX MODEL with XXX performance. A detailed implementation code can be find in the  `model_pipeline/src`/ folder.

### Deployment on AWS

This machine learning model has been trained and deployed on AWS, utilizing its powerful infrastructure and services. The use of AWS allows us to leverage scalable computing resources, efficient storage, and easy deployment options. By harnessing AWS capabilities, we ensure the model's availability and performance, enabling reliable predictions for loan default detection.

The architecture for our design is depicted in the graph below, showcasing the different components and their interactions.

[Insert Architecture Graph Image Here]

This diagram provides an overview of the key components and their connections within our system. It highlights the integration of AWS services, including data storage, model training, and deployment.

With this architecture in place, we are able to achieve reliable and efficient predictions for loan default detection, empowering organizations to make informed decisions and mitigate financial risks.


## Directory Structure
```
├── README.md                       <- You are here
├── model_app/                      <- Directory for streamlit application
│   ├── config/                     <- Directory for configuration files
│   │   └── logging/                <- Directory for python logging files
│   │       └── local.conf          <- Configuration file for python loggers
│   ├── Dockerfiles/                <- Directory for dockerfiles
│   │   ├── Dockerfile              <- Dockerfile for streamlit application  
│   │   └── Dockerfile.test         <- Dockerfile for unit testing
│   ├── src/                        <- Source data in python scripts for the project
│   │   └── aws_utils.py            <- Python script that interacts with AWS S3
│   ├── tests/                      <- Directory for running unit tests
│   │   └── test_app.py             <- Python script that tests the application functions
│   ├── app.py                      <- Python script that invokes streamlit application
├── └──requirements.txt             <- Text file for python package dependencies for running application
├── model_pipeline/                 <- Directory for streamlit application
│   ├── config/                     <- Directory for configuration files
│   │   ├── downloaded-config.yaml  <- yaml file for downloaded configuration
│   │   ├── logging_config.conf     <- Configuration file for python loggers 
│   │   └── piepline-config.yaml    <- yaml file for pipeline configuration
│   ├── Dockerfiles/                <- Directory for dockerfiles
│   │   ├── Dockerfile              <- Dockerfile for model pipeline
│   │   └── Dockerfile.UnitTest     <- Dockerfile for unit testing
│   ├── notebooks/                  <- Jupyter notebooks for EDA and modeling
│   │   ├── EDA_final.ipynb         <- Jupyter notebooks for EDA
│   │   └── Loan_Classification.ipynb <- Jupyter notebooks for modeling
│   ├── src/                        <- Source data in python scripts for the project
│   │   ├── acquire_data.py         <- Python script that acquires data from online repository 
│   │   ├── aws_utils.py            <- Python script that uploads artifacts to AWS S3 bucket
│   │   ├── create_dataset.py       <- Python script that creates structured dataset from raw data
│   │   ├── evaluate_performance.py <- Python script that evaluates model performance metrics
│   │   ├── score_model.py          <- Python script that scores model on test set
│   │   ├── Subsampling.py          <- Python script that takes samples of data
│   │   └── train_model.py          <- Python script that performs model training
│   ├── tests/                      <- Directory for running unit tests
│   │   ├── pipeline_modeling.py    <- Python script that tests modeling
│   │   └── main.py                 <- Python script that tests the pipeline
│   ├── pipeline_modeling.py        <- Python script that runs pipeline
│   └── main.py                     <- Python script that runs model
└── └──requirements.txt                <- Text file for python package dependencies for running application
```

## Environment Setup

#### AWS Credentials

##### Method 1: using a named profile from credentials file

First, `aws` CLI should be installed. Secondly, an aws profile needs to be configured as following:
```bash
aws config sso --profile "your-profile-name"
``` 
*Note: "your-profile-name" needs to be repalced with your actual aws profile name.*

Next, you may need to login to refresh credentials.
```bash
aws sso login --profile "your-profile-name"
```

Then, you can verify your identity with `sts`
```bash
aws sts get-caller-identity --profile "your-profile-name"
```

##### Method 2: Using AWS credential environment variables

The following two AWS environment variables are needed to be configured:
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`

To configure AWS credentials as environment variables, please run the following commands in the terminal.
```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
```  

To check if the credentials are set successfully, please run the following:
```bash
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

#### Python Requirements
* To run this project locally, you need to set up your python environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```
* To run this project on Docker, Python version 3.9 is required.

#### Packages Installation
* To run this project locally, please navigate to the root directory and run ```pip3 install requirements.txt```
* To run this project on Docker, installing packages in `requirements.txt` should be already specified when buidling the docker image (see below for details about Docker).


## Docker Instructions

### Docker for Model Pipeline 

#### Build the image

```bash
docker build -t cloud_loan . -f dockerfiles/Dockerfile
```

#### Run Pipeline

```bash
docker run \
-v $(pwd)/results:/app/runs \
-v $(pwd)/config:/app/config \
-v $(pwd)/results/log:/app/logs \
-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
-e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
cloud_loan
```

#### Run Unit Tests

Navigate to the `model_pipeline` directory and Use dockerfile named Dockerfile_UintTest to build a docker image for unit tests:

```bash
docker build -t loan_test . -f Dockerfiles/Dockerfile.UintTest
```

Run the build docker image to run the tests.

```bash
docker run loan_test
```
### Docker for Streamlit Application

#### Build the Docker image for application

```bash
docker build -f dockerfiles/Dockerfile -t loan-app .
```
This will build a docker iamge named `loan-app` for running the streamlit application.

#### Run the streamlit application

```bash
docker run -p 80:80 -e AWS_PROFILE="your-aws-profile" loan-app
```
OR
```bash
docker run -p 80:80 -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN loan-app
```
In both cases, the AWS credentials will be passed in for interacting with S3 bucket, and the command `streamlit run app.py` will be executed in the docker container to run the model pipeline.

#### Build the Docker image for tests

```bash
docker build -f dockerfiles/Dockerfile.test -t loan-test .
```
This will build a docker iamge named `loan-test` for running unit tests inside `tests/` directory.

#### Run the tests

```bash
docker run -e AWS_PROFILE="your-aws-profile" loan-test
```
OR
```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN app-test
```

The command `python3 -m pytest` will be executed in the docker container to run the unit tests inside `tests/` directory.