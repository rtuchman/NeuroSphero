import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import itertools


class NeuroLearnCNN(object):

    def __init__(self):

        self.classifier = Sequential()

        # First layer
        self.classifier.add(Convolution2D(32, (3, 3), input_shape=(121, 10, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Second layer
        self.classifier.add(Convolution2D(32, (3, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Third layer
        self.classifier.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Forth layer
        self.classifier.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

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
        self.classifier.add(Dense(activation='relu', input_dim=121, units=90, kernel_initializer='uniform'))

        # Adding the second hidden layer
        self.classifier.add(Dense(units=90, kernel_initializer="uniform", activation='relu'))

        # Adding the third hidden layer
        self.classifier.add(Dense(units=90, kernel_initializer="uniform", activation='relu'))

        # Adding the output layer
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))

        # Compiling the ANN
        optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        self.classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def data_preprocessing(self):
        # Importing the dataset
        dataset = pd.read_csv(r'neuro_data.csv')
        X = dataset.iloc[:, 1:122].values
        y = dataset.iloc[:, 122:].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Feature Scaling
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def train(self):

        # to view tensorboard after training open command line in project's folder and run:
        # tensorboard --logdir ./ --host localhost --port 8088
        # than open in your browser: http://localhost:8088
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)
        # Fitting the ANN to the Training set
        self.history = self.classifier.fit(self.X_train,self.y_train, validation_split=0.33,
                                           batch_size=10, nb_epoch=30, callbacks=[tbCallBack])

    def predict(self):
        # Predicting the Test set results
        self.y_pred = self.classifier.predict(self.X_test)
        self.y_pred = (self.y_pred > 0.9)

        # Making the Confusion Matrix
        y_test_non_category = [np.argmax(t) for t in self.y_test]
        y_predict_non_category = [np.argmax(t) for t in self.y_pred]
        self.cm = confusion_matrix(y_test_non_category, y_predict_non_category)


    def plot_confusion_matrix(self, cm, target_names, title='Confusion matrix', cmap=None, normalize=True):

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig('Confusion Matrix.jpg')




if __name__ == "__main__":
    model = NeuroLearnANN()
    model.data_preprocessing()
    model.train()
    model.predict()
    model.plot_confusion_matrix(model.cm, ['Memory game', 'Meditate', 'Write with weak hand'])
    model.classifier.save('NeuroClassifier.h5')

    print('Done!')

