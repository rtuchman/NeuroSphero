from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


import warnings
warnings.filterwarnings("ignore")
import tensorboard

class NeuroLearn:

    def __init__(self):


        self.classifier = Sequential()

        # First layer
        self.classifier.add(Convolution2D(32, (3, 3), input_shape=(121, 10, 3), activation='relu', padding="same"))  #CHANGE IMAGE SIZE
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Second layer
        self.classifier.add(Convolution2D(32, (3, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Third layer
        self.classifier.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Forth layer
        self.classifier.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        # Fifth layer
        #self.classifier.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        #self.classifier.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        # Output layers
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=128, activation='relu'))
        #self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(units=3, activation='softmax'))

        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def data_preprocessing(self):

        self.train_datagen = ImageDataGenerator(rescale=1./255,  # scale pixels [0-255] to [0,1]
                                           shear_range=0,
                                           zoom_range=0,
                                           horizontal_flip=False)

        self.test_datagen = ImageDataGenerator(rescale=1./255)  # scale pixels [0-255] to [0,1]

        self.training_set = self.train_datagen.flow_from_directory(
            r'C:\Users\owner\Desktop\NeuroSreer Project\dataset\train_data',
            target_size=(121, 10),
            batch_size=20,
            class_mode='categorical')

        self.test_set = self.test_datagen.flow_from_directory(
            r'C:\Users\owner\Desktop\NeuroSreer Project\dataset\test_data',
            target_size=(121, 10),
            batch_size=10,
            class_mode='categorical')


    def train(self):

        #tensorboard("logs/run_a")
        self.history = self.classifier.fit_generator(self.training_set,
                                      samples_per_epoch=384,
                                      nb_epoch=50,
                                      verbose=2,
                                      #callbacks=callback_tensorboard("logs/run_a"),
                                      validation_data=self.test_set,
                                      nb_val_samples=33)



if __name__ == "__main__":
    model = NeuroLearn()
    model.data_preprocessing()
    model.train()
    model.classifier.save_weights('first_try.h5')

    print('Done!')

