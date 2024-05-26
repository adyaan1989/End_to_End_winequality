# End_to_End_winequality Project

## Workflows to be used after the creation of the templates

1. Update config/config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the src/winequality/entity/__init__.py
5. Update the src/config/configuration.py
6. Update the src/components (multiple py files with steps)
7. Update the src/pipeline (multiple py files with steps) 
8. Update the main.py
9. Update the app.py

# Steps for running configuration of the application 

Clone the repository

```bash
https://github.com/entbappy/End-to-end-Machine-Learning-Project-with-MLflow
```

### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n venvWq python=3.9 -y
```

```bash
conda activate venvWq
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


## Step - 3 MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub

[dagshub](https://dagshub.com/)

# setup on the model evaluation file
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/singh.shanu1988/End_to_End_winequality.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="singh.shanu1988"
os.environ["MLFLOW_TRACKING_PASSWORD"]="85d3b3428bfdf23ee7d518a37c50b190e4e54914"


Run this on the git Bash to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/singh.shanu1988/End_to_End_winequality.mlflow \

export MLFLOW_TRACKING_USERNAME=singh.shanu1988 \

export MLFLOW_TRACKING_PASSWORD=85d3b3428bfdf23ee7d518a37c50b190e4e54914 \

python script.py
```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	### Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	### Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com/winequality

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	### optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	### required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app



## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model



