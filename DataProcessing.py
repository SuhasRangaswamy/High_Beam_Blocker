import tensorflow as tf
#import tensorflow_datasets as tfds

class DP:
    def __init__(self):
        pass

    def normalize(self, input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask = tf.cast(input_mask, tf.float32) / 255.0
        return input_image, input_mask

    @tf.function
    def image_process_train(self, datapoint, normalize=True):
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(datapoint[0])
            input_mask = tf.image.flip_left_right(datapoint[1])
        else:
            input_image = datapoint[0]
            input_mask = datapoint[1]

        if normalize:
            input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask

    def image_process_test(self, datapoint, normalize=True):
        if normalize:
            input_image, input_mask = self.normalize(datapoint[0], datapoint[1])

        return input_image, input_mask



