import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import \
	fmin_l_bfgs_b  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
# tf.compat.v1.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "taj_mahal.png"
STYLE_IMG_PATH = "starry_night.jpg"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1  # Alpha weight.
STYLE_WEIGHT = 1.0  # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3

# =============================<Helper Functions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


def deprocessImage(img):
	img[:, :, 0] += 103.939
	img[:, :, 1] += 116.779
	img[:, :, 2] += 123.68
	img = img[:, :, ::-1]
	img = np.clip(img, 0, 255).astype('uint8')
	return img


def gramMatrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram


# ========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
	channels = 3
	return (K.sum(K.square(gramMatrix(style) - gramMatrix(gen)))) / (4. * (channels ** 2)
																	* (CONTENT_IMG_H * CONTENT_IMG_W ** 2))


def contentLoss(content, gen):
	return K.sum(K.square(gen - content))


def totalLoss(x):
	a = K.square(
		x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
	b = K.square(
		x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])

	return K.sum(K.pow(a + b, 1.25))


# =========================<Pipeline Functions>==================================

def getRawData():
	print("   Loading images.")
	print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
	print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
	cImg = load_img(CONTENT_IMG_PATH)
	tImg = cImg.copy()
	sImg = load_img(STYLE_IMG_PATH)
	print("      Images have been loaded.")
	return (
		(cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
	img, ih, iw = raw
	img = img_to_array(img)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		img = imresize(img, (ih, iw, 3))
	img = img.astype("float64")
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)
	return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''


def styleTransfer(content, style, test):
	print("   Building transfer model.")
	contentTensor = K.variable(content)
	styleTensor = K.variable(style)
	genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
	inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
	model = None  # TODO: implement.
	outputDict = dict([(layer.name, layer.output) for layer in model.layers])
	print("   VGG19 model loaded.")
	loss = 0.0
	styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
	contentLayerName = "block5_conv2"
	print("   Calculating content loss.")
	contentLayer = outputDict[contentLayerName]
	contentOutput = contentLayer[0, :, :, :]
	genOutput = contentLayer[2, :, :, :]
	loss += None  # TODO: implement.
	print("   Calculating style loss.")
	for layerName in styleLayerNames:
		loss += None  # TODO: implement.
	loss += None  # TODO: implement.
	# TODO: Setup gradients or use K.gradients().
	print("   Beginning transfer.")
	for i in range(TRANSFER_ROUNDS):
		print("   Step %d." % i)
		# TODO: perform gradient descent using fmin_l_bfgs_b.
		print("      Loss: %f." % tLoss)
		img = deprocessImage(x)
		saveFile = None  # TODO: Implement.
		# imsave(saveFile, img)   #Uncomment when everything is working right.
		print("      Image saved to \"%s\"." % saveFile)
	print("   Transfer complete.")


# =========================<Main>================================================

def main():
	print("Starting style transfer program.")

	content_image, style_image, test_image = getRawData()

	content = preprocessData(content_image)  # Content image.
	style = preprocessData(style_image)  # Style image.
	test = preprocessData(test_image)  # Transfer image.

	styleTransfer(content, style, test)

	print("Done. Goodbye.")


if __name__ == "__main__":
	main()
