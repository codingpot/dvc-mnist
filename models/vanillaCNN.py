import tensorflow as tf
import tensorflow.keras.layers as layers

"""
This SegmentationModel origin is keras code example.
Author: fchollet
Link : https://keras.io/examples/vision/mnist_convnet/
Refactoring : Eden Park
"""

class VanillaCNN:
	def __init__(self, image_size, num_classes):
		self.input_shape = image_size + (1,)
		self.num_classes = num_classes

	def build_model(self):
		inputs = tf.keras.Input(shape=self.input_shape)

		x = layers.Conv2D(32, kernel_size=(3, 3))(inputs)
		x = layers.Activation("relu")(x)
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = layers.Conv2D(64, kernel_size=(3, 3))(x)
		x = layers.Activation("relu")(x)
		x = layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = layers.Flatten()(x)
		x = layers.Dropout(0.5)(x)
		outputs = layers.Dense(num_classes, activation="softmax")(x)

		return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
	image_size = (28, 28)
	num_classes = 3

	model = VanillaCNN(image_size=image_size, num_classes=num_classes)
	model = model.build_model()
	print(model.summary())
