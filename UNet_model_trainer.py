import tensorflow as tf
#import tensorflow_datasets as tfds
from DataProcessing import DP
import matplotlib.pyplot as plt
import DataLoader
import UNet_model
import math
import os
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

inputImagesPath = "../datasets/HighBB/Images"
truthPath = "../datasets/HighBB/truth.json"
TrainedModelDir = "../TrainedModels/HBB"
ModelName = "Unet_HBB"

BATCH_SIZE = 2
#BUFFER_SIZE = 500
OUTPUT_CHANNELS = 2
BASE_LR_RATE = 0.001
MIN_LR_RATE = 0.00001
lrDeltaFactor = 0.8
lrDeltaStep = 100
EPOCHS = 100


def lr_scheduler(epoch):
    new_lr = BASE_LR_RATE
    decCount = math.floor(epoch/lrDeltaStep)
    for i in range(0, decCount):
        new_lr *= lrDeltaFactor
    if new_lr < MIN_LR_RATE:
        new_lr = MIN_LR_RATE
    tf.summary.scalar("Learning Rate", new_lr, epoch)
    return new_lr


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Load the dataset
print("Loading the dataset....")
dataset, dataset_len = DataLoader.loadDataset(inputImagesPath, truthPath)
print("Done")

# Split the dataset into train and test
print("Spliting the dataset....")
dataset_train, dataset_test = DataLoader.splitDataset(dataset)
x_train, y_train = DataLoader.seperateDatasets(dataset_train)
x_test, y_test = DataLoader.seperateDatasets(dataset_test)
print("Done")

# Randomly flip and normalize the images
print("Pre-processing the data....")
dp = DP()
x_train, y_train = dp.image_process_train([x_train, y_train])
x_test, y_test = dp.image_process_test([x_test, y_test])
print("Done")

STEPS_PER_EPOCH = dataset_len // BATCH_SIZE

print("Defining model....")
model = UNet_model.unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Done")

outputModelDir = TrainedModelDir + "/" + ModelName + "_" + str(BASE_LR_RATE) + "_" + str(EPOCHS) + "_" + str(BATCH_SIZE)

##### Defining callbacks #####
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath= outputModelDir + "/Checkpoints/checkpoint_{epoch}.h5",
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the 'val_loss' score has improved
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        # Stop training when 'val_loss' is no longer improving
        monitor='val_loss',
        # 'no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=45,
        verbose=1
    ),
]

fileWriter = tf.summary.create_file_writer(outputModelDir + "/tf_logs/scalars")
fileWriter.set_as_default()

##### Adding LR callback #####
lrCallback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, True)
callbacks.append(lrCallback)

##### Display callback #####


##### Creating output Directories #####
os.makedirs(outputModelDir, exist_ok=True)
os.makedirs(outputModelDir + "/Checkpoints", exist_ok=True)
os.makedirs(outputModelDir + "/tf_logs", exist_ok=True)

##### Training #####
print("Training....")
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=callbacks)
loss = history.history['loss']
val_loss = history.history['val_loss']
print("Done")

# Saving the trained model as tensorflow saved model
print("Saving the trained model....")
tf_path = outputModelDir + "/SavedModel"
if not os.path.exists(tf_path):
    os.mkdir(tf_path)
print("Exporting trained model as tensorflow model to ", tf_path)
tf.saved_model.save(model, tf_path)

