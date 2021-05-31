import tensorflow as tf
import tensorflow.keras.layers as layers

"""
This SegmentationModel origin is keras code example.
Author: fchollet
Link : https://keras.io/examples/vision/oxford_pets_image_segmentation/
Refactoring : Eden Park
"""

class SegmentationModel:
	def __init__(self, image_size, num_classes):
		self.input_shape = image_size + (3,)
		self.num_classes = num_classes

	def build_model(self):
		inputs = tf.keras.Input(shape=self.input_shape)

		### [First half of the network: downsampling inputs] ###

		# Entry block
		x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
		x = layers.BatchNormalization()(x)
		x = layers.Activation("relu")(x)

		previous_block_activation = x  # Set aside residual

		# Blocks 1, 2, 3 are identical apart from the feature depth.
		for filters in [64, 128, 256]:
			x = layers.Activation("relu")(x)
			x = layers.SeparableConv2D(filters, 3, padding="same")(x)
			x = layers.BatchNormalization()(x)

			x = layers.Activation("relu")(x)
			x = layers.SeparableConv2D(filters, 3, padding="same")(x)
			x = layers.BatchNormalization()(x)

			x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

			# Project residual
			residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
				previous_block_activation
			)
			x = layers.add([x, residual])  # Add back residual
			previous_block_activation = x  # Set aside next residual

		### [Second half of the network: upsampling inputs] ###

		for filters in [256, 128, 64, 32]:
			x = layers.Activation("relu")(x)
			x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
			x = layers.BatchNormalization()(x)

			x = layers.Activation("relu")(x)
			x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
			x = layers.BatchNormalization()(x)

			x = layers.UpSampling2D(2)(x)

			# Project residual
			residual = layers.UpSampling2D(2)(previous_block_activation)
			residual = layers.Conv2D(filters, 1, padding="same")(residual)
			x = layers.add([x, residual])  # Add back residual
			previous_block_activation = x  # Set aside next residual

		# Add a per-pixel classification layer
		outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

		# Define and return the model
		return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
	image_size = (160, 160)
	num_classes = 3

	model = SegmentationModel(image_size=image_size, num_classes=num_classes)
	model = model.build_model()
	print(model.summary())
