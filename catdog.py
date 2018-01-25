import discord
import Image_Classifier as ic
import numpy
import os
import random
import requests
import string
import tensorflow as tf
import tflearn as tfl
import urllib.request

from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from discord.ext import commands
from PIL import Image


description = '''This bot just helps you know if the picture you have has a cat or a dog,
 if the picture has something else it will just say if it looks like a cat or a dog.'''

bot = commands.Bot(command_prefix='?', description=description)
model = ic.buildModel()
model.load(ic.MODEL_FILE,  weights_only = True)

@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')

@bot.command()
async def classify(link : str):
    """Classifies if an image from a given url has a dog or a cat.
       Note, the comand doesnt suppot embeded images for now """
    try:
        filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        with open(filename, 'wb') as file: 
            file.write(requests.get(link, allow_redirects=True).content)
        img = Image.open(filename)
        os.remove(filename)
        img = numpy.asarray(img.resize(ic.IMAGE_SHAPE))[:,:,:3] / 255
        prediction = model.predict([img])[0]
        if prediction[0] > prediction[1]:
            predicted_name = 'cat'
        else:
            predicted_name = 'dog'
        await bot.say("I think it's a {}..".format(predicted_name))
    except :
        await bot.say("Ups... Something went wrong")
        raise


bot.run('token')
