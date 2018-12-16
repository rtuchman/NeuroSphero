import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras import optimizers
from keras.callbacks import TensorBoard
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")



"""THIS MODULE RUNS ON PYTHON 3.6"""


class NeuroLearnANN(object):

    def __init__(self):
        # Initialising the ANN
        self.classifier = Sequential()

        # Adding the input layer and the first hidden layer
        self.classifier.add(Dense(activation='relu', input_dim=121, units=135, kernel_initializer='glorot_uniform'))

        # kernel_regularizer=regularizers.l1(0.001)

        # Adding the second hidden layer
        self.classifier.add(Dropout(0.1))
        self.classifier.add(Dense(units=135, kernel_initializer="glorot_uniform", activation='relu'))

        # Adding the third hidden layer
        self.classifier.add(Dropout(0.1))
        self.classifier.add(Dense(units=135, kernel_initializer="glorot_uniform", activation='relu'))

        # Adding the output layer
        self.classifier.add(Dropout(0.1))
        self.classifier.add(Dense(units=4, kernel_initializer='glorot_uniform', activation='softmax'))

        # Compiling the ANN
        optimizer = optimizers.Adam(lr=0.0008, beta_1=0.77, beta_2=0.98, epsilon=1e-8, decay=0.0, amsgrad=False)
        self.classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def data_preprocessing(self):
        # Importing the dataset
        dataset = pd.read_csv(r'neuro_data.csv')
        X = dataset.iloc[:, 1:122].values
        y = dataset.iloc[:, 122:].values

        # Splitting the dataset into the Training set and Test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=22)

        # Feature Scaling
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def train(self):

        # to view tensorboard after training open anaconda prompt in project's folder and run:
        # tensorboard --logdir ./ --host localhost --port 8088
        # now open in your browser: http://localhost:8088
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)


        # Fitting the ANN to the Training set
        # you may use history to view accuracy
        self.history = self.classifier.fit(self.X_train, self.y_train,
                                           validation_data=(self.X_test, self.y_test),
                                           batch_size=10, nb_epoch=100, shuffle=True,
                                           callbacks=[tbCallBack])

        self.save_graphs()

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


    def save_graphs(self):
        epochs = self.history.epoch
        plt.plot(epochs, self.history.history['val_acc'], label='val_acc')
        plt.plot(epochs, self.history.history['acc'], label='train_acc')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.savefig('Accuracy')
        plt.close()
        plt.plot(epochs, self.history.history['val_loss'], label='val_loss')
        plt.plot(epochs, self.history.history['loss'], label='train_loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.savefig('Loss')
        plt.close()





if __name__ == "__main__":
    model = NeuroLearnANN()
    model.data_preprocessing()
    model.train()
    model.predict()
    model.plot_confusion_matrix(model.cm, ['Memory game', 'Meditate', 'Write with weak hand', 'Happy music (dancing)'])
    model.classifier.save('NeuroClassifier.h5')


    print('Done!')

