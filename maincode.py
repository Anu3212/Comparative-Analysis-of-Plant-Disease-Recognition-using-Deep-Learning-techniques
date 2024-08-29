# Dowload the packages
!pip install split-folders

from tensorflow.keras.applications import MobileNetV2

import numpy as np
import pandas as pd
import os, os.path
import splitfolders
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

import keras.backend as K
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.applications import MobileNetV2  # Import MobileNetV2

matplotlib.style.use('ggplot')
%matplotlib inline


# Set some default variables
DATA_DIR = '/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
BATCH_SIZE = 128
EPOCHS = 20
IMAGE_SHAPE = (224, 224)


import keras.backend as K

def f1_macro(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives + K.epsilon())    
        return recall 

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives + K.epsilon())
        return precision 

    def accuracy_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        total_samples = K.cast(K.shape(y_true)[0], dtype='float32')

        accuracy = (true_positives + true_negatives) / total_samples
        return accuracy

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    accuracy = accuracy_m(y_true, y_pred)
    
    return {
        'f1_macro': 2 * ((precision * recall) / (precision + recall + K.epsilon())),
        'accuracy': accuracy
    }


# Check the content

pairs = list()
number = list()

for directory in os.listdir(path=DATA_DIR):
    columns = directory.split('___')
    columns.append(directory)
    
    sub_path = DATA_DIR + '/' + directory
    columns.append(len([name for name in os.listdir(path=sub_path)]))
    
    pairs.append(columns)
    
pairs = pd.DataFrame(pairs, columns=['Plant', 'Disease', 'Directory', 'Files'])
pairs.sort_values(by='Plant')


# Make a directory images
try:
    os.mkdir('images')

# Make subdirectories train, val, test
    os.mkdir(os.path.join('images', 'train'))
    os.mkdir(os.path.join('images', 'val'))
    os.mkdir(os.path.join('images', 'test'))
except:
    pass



# Split the data into folders
splitfolders.ratio(DATA_DIR,output = "images",seed = 42,ratio = (0.80,0.10,0.10))


# Remove the directories we saved before

TRAIN_PATH = "./images/train"
VAL_PATH = "./images/val"
TEST_PATH  = "./images/test"
PATHS = [TRAIN_PATH, VAL_PATH, TEST_PATH]


# Generate batches of tensor image data with real-time data augmentation

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_gen = datagen.flow_from_directory(directory = TRAIN_PATH, 
                                          class_mode="categorical",
                                          target_size = IMAGE_SHAPE,
                                          batch_size = BATCH_SIZE,
                                          color_mode='rgb',
                                          seed = 1234,
                                          shuffle = True)

val_gen = datagen.flow_from_directory(directory = VAL_PATH, 
                                          class_mode="categorical",
                                          target_size = IMAGE_SHAPE,
                                          batch_size = BATCH_SIZE,
                                          color_mode='rgb',
                                          seed = 1234,
                                          shuffle = True)

test_gen = datagen.flow_from_directory(directory = TEST_PATH, 
                                          class_mode="categorical",
                                          target_size = IMAGE_SHAPE,
                                          batch_size = BATCH_SIZE,
                                          color_mode='rgb',
                                          shuffle = False)


try:# Make a directory for models
    os.mkdir('models')

# Make subdirectories train, val, test
    os.mkdir(os.path.join('models', 'first_version'))
except:
    pass

# Here we create checkpoint for the first model
CHECKPOINT_PATH_MODEL_FIRST = "./models/first_version"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH_MODEL_FIRST,
                                      monitor='val_loss',
                                      save_best_only=True)

# Set early stopping for 2 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience = 2, restore_best_weights=True)


# Import MobileNetV2

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Use pre-trained MobileNetV2 model as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier on top of the pre-trained model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(38, activation='softmax')(x)

# Create the model
mobilenet_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
mobilenet_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy', f1_macro])

# Display the model summary
mobilenet_model.summary()


mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,  # Number of epochs with no improvement to wait
    verbose=1,
    restore_best_weights=True  # Restore the best weights when stopping
)


history = mobilenet_model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs= 20,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    verbose=1,
    callbacks=[early_stopping]  # Remove or comment out this line
)



# Evaluate the model on the test dataset using test_gen
test_loss, test_accuracy = mobilenet_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


import matplotlib.pyplot as plt

# Assuming you have a 'history' object containing accuracy and validation accuracy values

# Extract accuracy and validation accuracy from the 'history' object
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create a list of epoch numbers
epochs = range(1, len(accuracy) + 1)

# Plot accuracy
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('mobilenet')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
# Evaluate the model on the test dataset using test_gen
test_loss, test_accuracy = mobilenet_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict labels for the test data
predictions = mobilenet_model.predict(test_gen)
predicted_labels = np.argmax(predictions, axis=1)

# Get the true labels for the test data
true_labels = test_gen.classes

# Calculate confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(confusion)

# Define the number of classes in your dataset (replace with the actual number)
num_classes = 38

# Calculate precision, recall, and F1 score for each class
class_precision = precision_score(true_labels, predicted_labels, average=None)
class_recall = recall_score(true_labels, predicted_labels, average=None)
class_f1_score = f1_score(true_labels, predicted_labels, average=None)

# Calculate macro-average precision, recall, and F1 score
macro_precision = np.mean(class_precision)
macro_recall = np.mean(class_recall)
macro_f1_score = np.mean(class_f1_score)

# Calculate micro-average precision, recall, and F1 score
micro_precision = precision_score(true_labels, predicted_labels, average='micro')
micro_recall = recall_score(true_labels, predicted_labels, average='micro')
micro_f1_score = f1_score(true_labels, predicted_labels, average='micro')

print('Class-wise Metrics:')
for i in range(num_classes):
    print(f'Class {i + 1}: Precision = {class_precision[i]:.4f}, Recall = {class_recall[i]:.4f}, F1 Score = {class_f1_score[i]:.4f}')

print(f'Macro-average Precision = {macro_precision:.4f}, Recall = {macro_recall:.4f}, F1 Score = {macro_f1_score:.4f}')
print(f'Micro-average Precision = {micro_precision:.4f}, Recall = {micro_recall:.4f}, F1 Score = {micro_f1_score:.4f}')


import numpy as np
from sklearn.metrics import classification_report

# Make predictions on the test dataset
predictions = mobilenet_model.predict(test_gen, steps=len(test_gen))

# Get true labels directly from the test generator
true_labels = test_gen.classes

# Convert predicted probabilities to class indices
predicted_labels = np.argmax(predictions, axis=1)

# Generate classification report
report = classification_report(true_labels, predicted_labels, output_dict=True)

# Extract overall metrics
overall_metrics = report['weighted avg']

# Print overall metrics
print("Overall Metrics:")
print(f"Precision: {overall_metrics['precision']:.4f}")
print(f"Recall: {overall_metrics['recall']:.4f}")
print(f"F1-score: {overall_metrics['f1-score']:.4f}")


# Plot validation accuracy vs. epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('mobilenet')
plt.legend()
plt.grid()
plt.show()


#RECOGNITION FROM GIVEN INPUT
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict probabilities for each class for an input image
def predict_probabilities(image_path):
    img_array = preprocess_image(image_path)
    predictions = mobilenet_model.predict(img_array)[0]
    return predictions

# Function to display the input image along with the predicted probabilities for each class
def display_image_with_probabilities(image_path):
    probabilities = predict_probabilities(image_path)
    img = image.load_img(image_path, target_size=(224, 224))

    plt.figure(figsize=(6, 8))
    plt.imshow(img)
    plt.axis('off')

    if np.max(probabilities) > 0.5:  # If the highest predicted probability is greater than 0.5, consider it as diseased
        predicted_label_index = np.argmax(probabilities)
        predicted_disease = list(train_gen.class_indices.keys())[predicted_label_index]
        plt.text(0, -20, f"The leaf is Recognized as Diseased: {predicted_disease}", fontsize=12, color='red')
    else:
        plt.text(0, -20, "The leaf is Recognized as Healthy", fontsize=12, color='green')

    plt.text(0, -40, "Predicted Probabilities:", fontsize=12, color='black')
    for i, prob in enumerate(probabilities):
        plt.text(0, -60-20*i, f"Class {i}: {prob:.4f}", fontsize=12, color='black')
    
    plt.show()

# Path to the input image
image_path = "/kaggle/input/new-plant-diseases-dataset/test/test/TomatoHealthy1.JPG"  # Replace with the path to your image

# Check if the image file exists
if not os.path.isfile(image_path):
    print("Error: The specified image file does not exist.")
else:
    # Display input image along with predicted probabilities for each class
    display_image_with_probabilities(image_path)

# Load ground truth labels for the dataset
ground_truth_labels = ['Diseased', 'Healthy']  # Replace with your actual ground truth labels

# Predict whether the input image is healthy or diseased
predicted_probabilities = predict_probabilities(image_path)
predicted_label = "Diseased" if np.max(predicted_probabilities) > 0.5 else "Healthy"

# Check if the prediction matches the ground truth label for the input image
input_image_ground_truth_label = 'Diseased'  # Replace with the ground truth label for the input image
is_prediction_correct = predicted_label == input_image_ground_truth_label

# Calculate accuracy for the input image recognition
input_image_accuracy = 1 if is_prediction_correct else 0

# Print accuracy for the input image recognition
print(f"Accuracy for recognition of the input image: {input_image_accuracy}")


# Set some default variables
DATA_DIR = '/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
BATCH_SIZE = 128
EPOCHS = 20  # <-- Define the number of epochs here
IMAGE_SHAPE = (224, 224)


# RESNET MODEL WITH 152 LAYERS

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet152V2

# Define the ResNet152V2 model
def create_resnet_model(input_shape=(224, 224, 3), num_classes=38):
    # Load pre-trained ResNet152V2 model
    base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classifier on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the model
    resnet_model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    return resnet_model

# Instantiate the ResNet152V2 model
resnet152v2_model = create_resnet_model(input_shape=(224, 224, 3), num_classes=38)

# Display the model summary
resnet152v2_model.summary()

# Train the model (assuming you have train_gen and val_gen already defined)
history = resnet152v2_model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=20,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    verbose=1,
    callbacks=[early_stopping]  # Use early stopping callback based on accuracy
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = resnet152v2_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot validation accuracy vs. epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Epochs')
plt.legend()
plt.grid()
plt.show()

# Predict labels for the test data
predictions = resnet152v2_model.predict(test_gen)
predicted_labels = np.argmax(predictions, axis=1)

# Get the true labels for the test data
true_labels = test_gen.classes

# Calculate confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(confusion)

# Print classification report
print('Classification Report:')
print(classification_report(true_labels, predicted_labels))

# Calculate precision, recall, and F1 score for each class
class_precision = precision_score(true_labels, predicted_labels, average=None)
class_recall = recall_score(true_labels, predicted_labels, average=None)
class_f1_score = f1_score(true_labels, predicted_labels, average=None)

# Calculate macro-average precision, recall, and F1 score
macro_precision = np.mean(class_precision)
macro_recall = np.mean(class_recall)
macro_f1_score = np.mean(class_f1_score)

print('Class-wise Metrics:')
for i in range(num_classes):
    print(f'Class {i + 1}: Precision = {class_precision[i]:.4f}, Recall = {class_recall[i]:.4f}, F1 Score = {class_f1_score[i]:.4f}')

print(f'Macro-average Precision = {macro_precision:.4f}, Recall = {macro_recall:.4f}, F1 Score = {macro_f1_score:.4f}')


# METRICS FOR RESNET152

import numpy as np
from sklearn.metrics import classification_report

try:# Make predictions on the test dataset
    predictions = resnet152v2_model.predict(test_gen, steps=len(test_gen))

# Get true labels directly from the test generator
    true_labels = test_gen.classes

# Convert predicted probabilities to class indices
    predicted_labels = np.argmax(predictions, axis=1)

# Generate classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)

# Extract overall metrics
    overall_metrics = report['weighted avg']

# Print overall metrics
    print("Overall Metrics:")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1-score: {overall_metrics['f1-score']:.4f}")
except:
    pass


import matplotlib.pyplot as plt

# Assuming you have a 'history' object containing accuracy and validation accuracy values

# Extract accuracy and validation accuracy from the 'history' object
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create a list of epoch numbers
epochs = range(1, len(accuracy) + 1)

# Plot accuracy
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Resnet')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()


# Function to predict disease label for an input image using RESNet model
def predict_disease_resnet(image_path):
    img_array = preprocess_image(image_path)
    predictions = resnet152v2_model.predict(img_array)  
    disease_labels = ['Apple___healthy', 'Apple___Cedar_apple_rust', 'Apple___Black_rot', 'Apple___Apple_scab', 
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___healthy',
                      'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___Black_rot',
                      'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                      'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Raspberry___healthy',
                      'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
                      'Tomato___Leaf_Mold', 'Tomato___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot', 'Tomato___Bacterial_spot',
                      'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___Late_blight']

    predicted_label_index = np.argmax(predictions)
    predicted_label = disease_labels[predicted_label_index]
    accuracy = predictions[0][predicted_label_index]

    return predicted_label, accuracy

# Function to display the input image along with the predicted disease label using RESNET model
def display_image_with_label_resnet(image_path):
    predicted_disease, accuracy = predict_disease_resnet(image_path)
    img = image.load_img(image_path, target_size=(224, 224))

    plt.figure(figsize=(6, 8))
    plt.imshow(img)
    plt.axis('off')

    plt.text(0, -20, f"recognized as: {predicted_disease.split('___')[1] if not predicted_disease.endswith('healthy') else 'Healthy'}", 
             fontsize=12, color='red')
    plt.text(0, -40, f"Accuracy: {accuracy*100:.2f}%", fontsize=12, color='blue')
    plt.show()

# Path to the input image
image_path = "/kaggle/input/new-plant-diseases-dataset/test/test/TomatoHealthy1.JPG"  

# Check if the image file exists
if not os.path.isfile(image_path):
    print("Error: The specified image file does not exist.")
else:
    # Display input image along with predicted disease label  model
    display_image_with_label_resnet(image_path)


# NASNet model with 1000+ layers

from tensorflow.keras.applications import NASNetLarge
# Create NASNetLarge model
base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier on top of the pre-trained model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(38, activation='softmax')(x)

# Create the model
nasnet_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
nasnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Display the model summary
nasnet_model.summary()

# Set early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = nasnet_model.fit(train_gen,
                           steps_per_epoch=len(train_gen),
                           epochs=20,
                           validation_data=val_gen,
                           validation_steps=len(val_gen),
                           verbose=1,
                           callbacks=[early_stopping])

# Evaluate the model on the test dataset
test_loss, test_accuracy = nasnet_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


#metrics for nasnet model

import numpy as np
from sklearn.metrics import classification_report

try:# Make predictions on the test dataset
    predictions = nasnet_model.predict(test_gen, steps=len(test_gen))

# Get true labels directly from the test generator
    true_labels = test_gen.classes

# Convert predicted probabilities to class indices
    predicted_labels = np.argmax(predictions, axis=1)

# Generate classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)

# Extract overall metrics
    overall_metrics = report['weighted avg']

# Print overall metrics
    print("Overall Metrics:")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1-score: {overall_metrics['f1-score']:.4f}")
except:
    pass


# Function to predict disease label for an input image using nasNet model
def predict_disease_nasnet(image_path):
    img_array = preprocess_image(image_path)
    predictions = nasnet_model.predict(img_array)  
    disease_labels = ['Apple___healthy', 'Apple___Cedar_apple_rust', 'Apple___Black_rot', 'Apple___Apple_scab', 
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___healthy',
                      'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___Black_rot',
                      'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                      'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Raspberry___healthy',
                      'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
                      'Tomato___Leaf_Mold', 'Tomato___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot', 'Tomato___Bacterial_spot',
                      'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___Late_blight']

    predicted_label_index = np.argmax(predictions)
    predicted_label = disease_labels[predicted_label_index]
    accuracy = predictions[0][predicted_label_index]

    return predicted_label, accuracy

# Function to display the input image along with the predicted disease label using nasnet
def display_image_with_label_nasnet(image_path):
    predicted_disease, accuracy = predict_disease_nasnet(image_path)
    img = image.load_img(image_path, target_size=(224, 224))

    plt.figure(figsize=(6, 8))
    plt.imshow(img)
    plt.axis('off')

    plt.text(0, -20, f"recognized as: {predicted_disease.split('___')[1] if not predicted_disease.endswith('healthy') else 'Healthy'}", 
             fontsize=12, color='red')
    plt.text(0, -40, f"Accuracy: {accuracy*100:.2f}%", fontsize=12, color='blue')
    plt.show()

# Path to the input image
image_path = "/kaggle/input/new-plant-diseases-dataset/test/test/TomatoHealthy1.JPG" 
# Check if the image file exists
if not os.path.isfile(image_path):
    print("Error: The specified image file does not exist.")
else:
    # Display input image along with predicted disease label using NASNet model
    display_image_with_label_nasnet(image_path)


import numpy as np
from sklearn.metrics import classification_report

try:# Make predictions on the test dataset
    predictions = sasnet_model.predict(test_gen, steps=len(test_gen))

# Get true labels directly from the test generator
    true_labels = test_gen.classes

# Convert predicted probabilities to class indices
    predicted_labels = np.argmax(predictions, axis=1)

# Generate classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)

# Extract overall metrics
    overall_metrics = report['weighted avg']

# Print overall metrics
    print("Overall Metrics:")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1-score: {overall_metrics['f1-score']:.4f}")
except:
    pass


import matplotlib.pyplot as plt

# Define the models and their corresponding metrics
models = ['MobileNet', 'AlexNet', 'ResNet152V2','NASNet','EfficientNet','SASNet']
precisions = [0.9595, 0.0008, 0.9712, 0.9207, 0.0008, 0.3860]
recalls = [0.9588, 0.0287, 0.9703, 0.9197, 0.0287, 0.3823]
f1_scores = [0.9586, 0.0016, 0.9704, 0.9188, 0.0016, 0.3479]

# Plot precision for each model
plt.figure(figsize=(7, 4))
plt.bar(models, precisions, label='Precision', color='pink')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Precision Comparison')
plt.xticks(rotation=45)
plt.show()

# Plot recall for each model
plt.figure(figsize=(7, 4))
plt.bar(models, recalls, label='Recall', color='red')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Recall Comparison')
plt.xticks(rotation=45)
plt.show()

# Plot F1-score for each model
plt.figure(figsize=(7, 4))
plt.bar(models, f1_scores, label='F1-score', color='orange')
plt.xlabel('Models')
plt.ylabel('F1-score')
plt.title('F1-score Comparison')
plt.xticks(rotation=45)
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize accuracy graph
def visualize_accuracy(accuracies, model_names):
    plt.figure(figsize=(7, 4))
    x = np.arange(len(accuracies))
    plt.bar(x, accuracies, color=['violet', 'violet','violet','violet','violet','violet'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(x, model_names)
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1 for percentage accuracy
    plt.show()

# Calculate accuracies for AlexNet and MobileNet models (You need to have the accuracies values)
alexnet_accuracy = 0.02  # Replace with the actual accuracy for AlexNet
mobilenet_accuracy = 0.95 # Replace with the actual accuracy for MobileNet
resnet_acc = 0.97
Nasnet_acc = 0.91
Efficientnet_acc = 0.028
SASNet_acc = 0.38

# Visualize accuracies
accuracies = [alexnet_accuracy, mobilenet_accuracy, resnet_acc, Nasnet_acc, Efficientnet_acc, SASNet_acc]
model_names = ['AlexNet', 'MobileNet', 'ResNet152', 'NASNet', 'EfficientNet','SASNet']
visualize_accuracy(accuracies, model_names)

#Alexnet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the input shape for the images
input_shape = (224, 224, 3)

# Define the AlexNet model
alexnet_model = Sequential([
    # Convolutional layers
    Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Fully connected layers
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(38, activation='softmax')  # Adjust this based on your number of classes
])

# Compile the model
alexnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
alexnet_model.summary()

# Train the model (assuming you have train_gen and val_gen already defined)
history = alexnet_model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=20,
    validation_data=val_gen,
    validation_steps=len(val_gen)
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = alexnet_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot validation accuracy vs. epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Epochs')
plt.legend()
plt.grid()
plt.show()


import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy - AlexNet Model')
plt.legend()
plt.grid()
plt.show()



from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Calculate predictions on the test set
y_pred = alexnet_model.predict(test_gen, steps=len(test_gen))
y_true = test_gen.classes

# Calculate precision, recall, F1-score, and accuracy
precision = precision_score(y_true, y_pred.argmax(axis=1), average='weighted')
recall = recall_score(y_true, y_pred.argmax(axis=1), average='weighted')
f1 = f1_score(y_true, y_pred.argmax(axis=1), average='weighted')
accuracy = accuracy_score(y_true, y_pred.argmax(axis=1))

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Accuracy: {accuracy:.4f}")



#googlenet
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load InceptionV3 pre-trained model without the top (fully connected) layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier on top of the pre-trained model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(38, activation='softmax')(x)

# Create the model
inception_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
inception_model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Display the model summary
inception_model.summary()

# Train the model
history = inception_model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=20,  # Adjust number of epochs as needed
    validation_data=val_gen,
    validation_steps=len(val_gen),
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model on the test dataset using test_gen
test_loss, test_accuracy = inception_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Evaluate the model on the test dataset
test_loss, test_accuracy = alexnet_model.evaluate(test_gen, steps=len(test_gen))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Plot validation accuracy vs. epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Epochs')
plt.legend()
plt.grid()
plt.show()
