# Spam or Ham

Using the [Kaggle's dataset of SMS spam](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) I will attempt to come up with a classifier to detect spam.

View the eda notebook to look at my eda and the model selection

# To use

## As API

- Install and activate conda environment
- launch api with Uvicorn
- Access at `localhost:8000/docs` or call api using curl

## As container

Build docker container with the following command at current directory
```
docker build -t <image name>:<tag> -f docker/simple_spam_classifer.dockerfile .

docker run -it <image name>:<tag> -p 8000
```

Access at `localhost:8000/docs`

## As URL

Available [here](https://simple-spam-classifier-4fhoqe4txq-as.a.run.app/docs)