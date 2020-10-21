import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
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


class Evaluator(object):

	def __init__(self, function):
		self.loss_value = None
		self.grads_values = None
		self.fetch_loss_and_grads = function

	def loss(self, x):
		assert self.loss_value is None
		x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
		outs = self.fetch_loss_and_grads([x])

		loss_value = outs[0]
		grad_values = outs[1].flatten().astype('float64')
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

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

	model = vgg19.VGG19(input_tensor=inputTensor, weights='imagenet', include_top=False)
	outputDict = dict([(layer.name, layer.output) for layer in model.layers])

	print("   VGG19 model loaded.")
	loss = 0.0
	styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
	contentLayerName = "block5_conv2"

	print("   Calculating content loss.")
	contentLayer = outputDict[contentLayerName]
	contentOutput = contentLayer[0, :, :, :]
	genOutput = contentLayer[2, :, :, :]
	loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)

	print("   Calculating style loss.")
	for layerName in styleLayerNames:

		contentLayer = outputDict[layerName]

		reference_features = contentLayer[1, :, :, :]
		genOutput = contentLayer[2, :, :, :]

		sloss = styleLoss(reference_features, genOutput)
		loss += (STYLE_WEIGHT / len(styleLayerNames)) * sloss

	loss += TOTAL_WEIGHT * totalLoss(genTensor)

	grads = K.gradients(loss, genTensor)[0]
	fetch_loss_and_grads = K.function([genTensor], [loss, grads])

	evaluator = Evaluator(fetch_loss_and_grads)

	result_prefix = CONTENT_IMG_PATH
	iterations = 10
	x = test.flatten()

	print("   Beginning transfer.")
	for i in range(iterations):

		x, loss_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

		print("      Loss: %f." % loss_val)

	img = x.copy().reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
	img = deprocessImage(img)
	fname = result_prefix + '_' + STYLE_IMG_PATH + '_' + "styled.png"
	imsave(fname, img)
	print("      Image saved to \"%s\"." % fname)
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
