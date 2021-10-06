# Code adapted from: https://www.tensorflow.org/tutorials/images/classification
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import csv
from datetime import datetime

def graph_classification (path_to_data, n, dataDir="graphs_filtered"):
    cwd=os.getcwd()
    if(cwd!=path_to_data):
        print("graph_classification called from wrong directory")
    else:
        # set necessary directories
        PATH = os.path.join(path_to_data, dataDir)
        train_dir = os.path.join(PATH, 'train')
        validation_dir = os.path.join(PATH, 'validation')
        train_pos_dir = os.path.join(train_dir, 'positive')  # directory with our training positive pictures
        train_neg_dir = os.path.join(train_dir, 'negative')  # directory with our training negative pictures
        train_ntr_dir = os.path.join(train_dir, 'neutral')  # directory with our training neutral pictures
        validation_pos_dir = os.path.join(validation_dir, 'positive')  # directory with our validation positive pictures
        validation_neg_dir = os.path.join(validation_dir, 'negative')  # directory with our validation negative pictures
        validation_ntr_dir = os.path.join(validation_dir, 'neutral')  # directory with our validation neutral pictures

        # assign count variables
        num_pos_tr = len(os.listdir(train_pos_dir))
        num_neg_tr = len(os.listdir(train_neg_dir))
        num_ntr_tr = len(os.listdir(train_ntr_dir))
        num_pos_val = len(os.listdir(validation_pos_dir))
        num_neg_val = len(os.listdir(validation_neg_dir))
        num_ntr_val = len(os.listdir(validation_ntr_dir))
        total_train = num_pos_tr + num_neg_tr + num_ntr_tr
        total_val = num_pos_val + num_neg_val + num_ntr_val

        # set constants
        batch_size = 64 #128
        epochs = 15
        IMG_HEIGHT = 150
        IMG_WIDTH = 150

        #prepare data
        train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
        validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

        #load images from disc
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                directory=train_dir,
                                                                shuffle=True,
                                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                class_mode='categorical')

        val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                    directory=validation_dir,
                                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                    class_mode='categorical')


        # create the model
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(3) # three neurons in output layer allow for three different categorization options
        ])

        # compile the model
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        # Model Summary
        model.summary()

        # train model
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
            #validation_steps=total_val // 16
        )

        # visualize training
        # ------------------
        #print(history.history)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs_range = range(epochs)

        plt.style.use('default')
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        #plt.show()
        
        # save data
        
        # Save the model
        model.save('models/correlation/graph_class_model.h5')

        # create classification_results directory
        dirPath = "./classification_results"
        try:
            os.mkdir(dirPath)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % dirPath)
        
        # find time and date for naming purposes
        now = datetime.now() # current date and time
        date_time = now.strftime("%m:%d:%Y_%H:%M:%S")

        fname="classification_results/learning_data_" + date_time + ".png"
        plt.savefig(fname)
        # ------------------

        # Generate Summary File
        # ------------------
        results = list()
        graphNum = "Graph_" + str(n)
        # create a file with the graphnames and correlations
        results.append(("Graph Number", "Epoch", "Train_Acc", "Val_Acc", "Train_Loss", "Val_Loss"))
        for i in epochs_range:
            results.append((graphNum, i+1, acc[i], val_acc[i], loss[i], val_loss[i]))

        with open('classification_results/classification_info.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(results)
        f.close()
