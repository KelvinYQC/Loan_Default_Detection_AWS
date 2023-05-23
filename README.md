# Loan Default Detection Model

This repository contains a loan default detection model trained on the data from the Kaggle competition "Home Credit Default Risk". The model aims to predict the likelihood of a loan applicant defaulting on their payments based on various features and historical data.

## Dataset

The dataset used in this project can be obtained from the following Kaggle competition link:
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Please download the dataset from the provided link and place it in the `data` directory before running the model.

## Model Development

The loan default detection model leverages machine learning models and utilizes various features such as applicant information, credit history, and previous loan details to make predictions. The model has been trained using XXX MODEL with XXX performance. A detailed implementation code can be find in the  `src`/ folder.

## Deployment on AWS

This machine learning model has been trained and deployed on AWS, utilizing its powerful infrastructure and services. The use of AWS allows us to leverage scalable computing resources, efficient storage, and easy deployment options. By harnessing AWS capabilities, we ensure the model's availability and performance, enabling reliable predictions for loan default detection.

The architecture for our design is depicted in the graph below, showcasing the different components and their interactions.

[Insert Architecture Graph Image Here]

This diagram provides an overview of the key components and their connections within our system. It highlights the integration of AWS services, including data storage, model training, and deployment.

With this architecture in place, we are able to achieve reliable and efficient predictions for loan default detection, empowering organizations to make informed decisions and mitigate financial risks.

## Docker

Build the image

```bash
docker build -t cloud_loan . -f dockerfiles/Dockerfile
```

### Run Pipeline

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

## Run Unit Tests

Navigate to the `2023-423-ycj6475-hw2` directory and Use dockerfile named Dockerfile_UintTest to build a docker image for unit tests:

```bash
docker build -t loan_test . -f dockerfiles/Dockerfile_UintTest
```

Run the build docker image to run the tests.

```bash
docker run loan_test
```
