# CatDogBot
Just a silly bot that tells you if you have a picture of a cat or a dog, it is written in python3.5.

This bot is my attempt at implementing my first image classifier, I was inspired by [this video](https://www.youtube.com/watch?v=cAICT4Al5Ow) and to train the model I used [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) from [Kaggle](https://www.kaggle.com). The implemented model is an AlexNet and I got its implementation from [tflearn examples](https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py), I only tweaked it to match the dataset, but I think it's a bit overkill.

# How to get it up and running
The process is not easy but I will try to explain.
1. install dependencies `pip install -r requirements.txt`
2. Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)
3. make sure you have more than 30 GB of storage space
4. run Image_Classifier `python3 Image_Classifier.py`
5. get a discord bot token and change the last line from CatBotDog.py `bot.run('token')` to your token
6. cross your fingers you didn't make a mistake
