from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

class NeuroLearn:

    def __init_(self):
        self.classifier = Sequential()

        # First layer
        self.classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))  #CHANGE IMAGE SIZE
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Second layer
        self.classifier.add(Convolution2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Third layer
        self.classifier.add(Convolution2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Output layers
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=128, activation='relu'))
        self.classifier.add(Dense(units=1, activation='softmax'))

        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def data_preprocessing(self, x_train, y_train):

        self.train_datagen = ImageDataGenerator(shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)

        self.training_set = self.train_datagen.flow_from_dataframe(target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')


    def train(self):


        self.classifier.fit_generator(self.training_set,
                                 samples_per_epoch=1,
                                 nb_epoch=25)


