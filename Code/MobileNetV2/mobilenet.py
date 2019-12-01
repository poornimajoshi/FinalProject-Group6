from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import numpy as np

# dimensions of our images
#img_width, img_height = 224, 224
img_width, img_height = 224, 224

weights_path = 'mobilenet_model.h5'
train_data_dir = '/home/ubuntu/Deep-Learning/FinalProject/data/cropped2/train/'
validation_data_dir = '/home/ubuntu/Deep-Learning/FinalProject/data/cropped2/test/'
test_data_dir = '/home/ubuntu/Deep-Learning/FinalProject/data/cropped2/validation/'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 8
model = applications.mobilenet.MobileNet(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_train_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

predict_size_train = int(math.ceil(nb_train_samples / batch_size))

bottleneck_features_train = model.predict_generator(
    generator, predict_size_train)

np.save('bottleneck_features_train_mob.npy', bottleneck_features_train)
generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_validation_samples = len(generator.filenames)

predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

bottleneck_features_validation = model.predict_generator(
    generator, predict_size_validation)

np.save('bottleneck_features_validation_mob.npy', bottleneck_features_validation)

datagen_top = ImageDataGenerator(rescale=1. / 255)
generator_top = datagen_top.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

# load the bottleneck features saved earlier
train_data = np.load('bottleneck_features_train_mob.npy')

# get the class lebels for the training data, in the original order
train_labels = generator_top.classes

# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)
generator_top = datagen_top.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_validation_samples = len(generator_top.filenames)

validation_data = np.load('bottleneck_features_validation_mob.npy')

validation_labels = generator_top.classes
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

model.save_weights(weights_path)

(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


prediction_logs = model.predict(validation_data)
prediction_output = np.argmax(prediction_logs, axis=-1)
print("Probs: ",prediction_logs[0])
print("Pred: ",prediction_output[0])
print("Length: ",len(prediction_output))

import pickle
with open('mobilenet_poornimajoshi_L.pickle', 'wb') as handle:
    pickle.dump(prediction_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('mobilenet_poornimajoshi_p.pickle', 'wb') as handle:
    pickle.dump(prediction_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("---Done---")