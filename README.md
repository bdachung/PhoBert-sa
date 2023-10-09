# Vietnamses Sentiment Analysis using phoBert

In this project, I use phoBert along with Huggingface for the task Vietnamese Sentiment Analysis of the competition [Hate Speech Detection on Social Networks](https://aihub.vn/competitions/9#learn_the_details)

# How to use

## First, clone the project

```
git clone https://github.com/bdachung/PhoBert-sa.git
```

## Training the model

Go to `src/models` and run

```
python main.py
```

## Make the prediction for the competition

Go to `src/models` and run the notebook `test.ipynb`, get the result in `data/submission` and submit.

## Custom for your dataset

You can go to `src/data/preprocessing.ipynb` and change some code to preprocess your data

OR you can put your preprocessed data in the folder `data/processed`. The dataset should include 2 columns, free_text (text) and label_id (label)

## I have a checkpoint-5000 [here](https://drive.google.com/drive/folders/1ZRkoL7LTjREvlcaNcmR8DMNiT0qkWIGK?usp=sharing)
