from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorboard

class NeuroLearnCNN(object):

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
        #self.classifier.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
        #self.classifier.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

        # Fifth layer
        ##self.classifier.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        ##self.classifier.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

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


class NeuroLearnANN(object):

    def __init__(self):
        # Initialising the ANN
        self.classifier = Sequential()

        # Adding the input layer and the first hidden layer
        # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))  # Deprecated
        self.classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))

        # Adding the second hidden layer
        # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))   # Deprecated
        self.classifier.add(Dense(units=6, kernel_initializer="uniform", activation='relu'))

        # Adding the output layer
        # classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))   # Deprecated
        self.classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        # Compiling the ANN
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def data_preprocessing(self):
        # Importing the dataset
        dataset = pd.read_csv(r'C:\Users\owner\Desktop\NeuroSreer Project\dataset\train_data\neuro_data.csv')
        X = dataset.iloc[:, 0:121].values
        y = dataset.iloc[:, 121:].values


        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)



    def train(self):

        # Fitting the ANN to the Training set
        self.classifier.fit(self.X_train, self.y_train, batch_size=10, nb_epoch=10)


    def predict(self):
        # Predicting the Test set results
        self.y_pred = self.classifier.predict(self.X_test)
        self.y_pred = (self.y_pred > 0.5)

        # Making the Confusion Matrix
        self.cm = confusion_matrix(self.y_test, self.y_pred)



if __name__ == "__main__":
    model = NeuroLearnANN()
    model.data_preprocessing()
    model.train()
    model.classifier.save_weights('first_try.h5')

    print('Done!')

