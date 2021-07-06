import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pylab as pl
import seaborn as sns

num_classes = 7
(w, h) = (260, 260)

efficientB2_model = Sequential()


efficientB2_model.add(EfficientNetB2(include_top=False, weights='imagenet', input_shape=(w,h,3), pooling='avg'))
# efficientB0_model.save("D:/Misaj/S3 MTech/Project/efficientB0_weights_tf_dim_ordering_tf_kernels_notop.h5")

# There is no need to simply save and load weights in each model.. Once downloaded weights can be used as it is again and again
# efficientB0_weights_path = 'D:/Misaj/S3 MTech/Project/efficientB0_weights_tf_dim_ordering_tf_kernels_notop.h5'
# efficientB0_model.add(EfficientNetB0(include_top=False, weights=efficientB0_weights_path, input_shape=(w,h,3), pooling='avg'))
efficientB2_model.add(Dense(num_classes, activation='softmax'))
efficientB2_model.summary()


# Say not to train first layer model. It is already trained
efficientB2_model.layers[0].trainable = False

# efficientB0_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy')
# efficientB2_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[tensorflow.keras.metrics.Precision(),'accuracy'])
efficientB2_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy'])


from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


image_size = 260
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# data_generator = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

train_generator = data_generator.flow_from_directory(
        'train_path',
        target_size=(image_size, image_size),
        batch_size=5,
        shuffle=False,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'val_path',
        target_size=(image_size, image_size),
        shuffle=False,
        class_mode='categorical')

test_generator = data_generator.flow_from_directory(
        'test_path',
        target_size=(image_size, image_size),
        shuffle=False,
        class_mode='categorical')

history = efficientB2_model.fit(
        train_generator,
        steps_per_epoch=690//32,
        epochs=60,
        validation_data=validation_generator,
        validation_steps=1)

efficientB2_model.save('models/model3.h5')

#test
metrics = efficientB2_model.evaluate(test_generator)
print('Loss, Precision, Recall, Accuracy :', metrics)
print ('precision', metrics[1])
print ('recall', metrics[2])
f_score = 2 * (( metrics[2] * metrics[1]) / (metrics[2] + metrics[1]))
print ('f1_score', f_score)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()


#print all labels for comparison
label_map = (test_generator.class_indices)
print('classes', label_map)

true_labels = test_generator.classes
print('labels', true_labels)

preds = efficientB2_model.predict(test_generator)
pred_labels = preds.argmax(axis=-1)
print('preds', pred_labels)



def showconfusionmatrix(cm, model_name):
    pl.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="rocket", xticklabels=label_map, yticklabels=label_map)
    pl.title('CM- ' + model_name)

def showmetrics(true, pred, model_name):
    accuracy = accuracy_score(true, pred)
    p, r, f, s = precision_recall_fscore_support(true, pred, average = 'weighted')
    cm = confusion_matrix(true, pred)
    
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP  
    FN = cm.sum(axis=1) - TP
    TN = [np.sum(cm), np.sum(cm), np.sum(cm), np.sum(cm), np.sum(cm), np.sum(cm), np.sum(cm)]
    TN = TN - (TP + FP + FN)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    TPR = np.average(TPR, axis=0)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    TNR = np.average(TNR, axis=0)
    
    print ("----Model----  ", model_name)
    print (cm)
    print ("accuracy  = ", accuracy)
    print ("precision = ", p)
    print ("recall    = ", r)
    print ("f1_score  = ", f)
    print ("TP       = ", TP)
    print ("FP       = ", FP)
    print ("FN       = ", FN)
    print ("TN       = ", TN)
    print ("Mean Sensitivity = ", TPR)
    print ("Mean Specificity = ", TNR)
    
    cm = confusion_matrix(true, pred, normalize='true')
    showconfusionmatrix(cm, model_name) 
    
    
showmetrics(true_labels, pred_labels, 'EfficientNetB2') 

from sklearn.metrics import classification_report
print('Classification Report')
print(classification_report(true_labels, pred_labels))
