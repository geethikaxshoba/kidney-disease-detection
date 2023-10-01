# kidney-disease-detection
This repository contains a Python-based machine learning project for the early detection of kidney disease. Kidney disease is a significant health concern, and early diagnosis can lead to better treatment outcomes. This project leverages machine learning techniques and data analysis to assist in the identification of potential kidney disease cases.

## Features
Data Preprocessing: Clean and preprocess the dataset for machine learning.
Machine Learning Models: Implement various classification algorithms to predict kidney disease.
Model Evaluation: Evaluate model performance using metrics such as accuracy, precision, and recall.
User-Friendly Interface: Develop a user-friendly interface for easy input and disease prediction.

## Importing Required Packages and Libraries.
```
import tensorflow as tf
from tensorflow.keras.preprocessing import image
#import tensorflow.keras.preprocessing
import keras
```
## Splitting the Dataset into Training , Validation and Testing Directories.
The aquired dataset was not divided into Training , Validation and Testing Directories. We used 'splitfolders' library and using the function 'splitfolders.ratio()' to divide the dataset as required.
```
import splitfolders
input_folder = 'Cataract Eye Detection Dataset/'
splitfolders.ratio(input_folder,output="Split Data Folder",
                   seed = 42, ratio = (0.7, 0.2, 0.1), group_prefix = None)
```
## Creating objects for ImageDataGenerator class and labeling them.
```
train = image.ImageDataGenerator(rescale = 1./255)
validation = image.ImageDataGenerator(rescale = 1./255)
```
Using 'flow_from_directory()' method to access the data folders and give labels to the data.
```
train_dataset = train.flow_from_directory(
    'Kidney Split Dataset/train/',
    batch_size = 20,
    target_size = (512,512),
    class_mode = 'categorical'
)
```
```
validation_dataset = validation.flow_from_directory(
    'Kidney Split Dataset/val/',
    batch_size = 20,
    target_size = (512,512),
    class_mode = 'categorical'
)
validation_dataset.class_indices
```
## Adding Callbacks to the Model.
```
class MyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('accuracy')>=0.99:
            print("\nTerminating the training as accuracy reached 95%")
            self.model.stop_training=True
```
## Defining the Convolutional Neural Network Architecture.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape = (512,512,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(input_shape = (512,512)),
    tf.keras.layers.Dense(256,activation = 'relu'),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(3,activation = 'softmax')
])
```
## Compiling the Model and Defining the loss function, optimizers, and metrics for prediction.
```
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics =['accuracy'])
```
## Instantiating the CallBack Object.
```
callback = MyCallbacks()
```
## Fitting the Data to the Model and Training it.
```
history = model.fit(train_dataset,
         validation_data = validation_dataset,
         epochs = 25,
          steps_per_epoch = 10,
         callbacks=[callback])
```
## Summary
```
model.save('MyModel.h5')
model = keras.models.load_model("MyModel.h5")
model.summary()
```
```
test_dataset = tf.keras.utils.image_dataset_from_directory(
          'Kidney Split Dataset/test/',
           shuffle=False,
           batch_size=1,
           image_size=(512,512))
L = ['Normal', 'Stone', 'Tumor']
def predict_image(img):
  img_4d=img.reshape(-1,512,512,3)
  prediction=model.predict(img_4d)[0]
  return {L[i]: float(prediction[i]) for i in range(3)}
import gradio as gr
image = gr.inputs.Image(shape=(512,512))
label = gr.outputs.Label(num_top_classes=3)

gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True',share = 'True')
```
## Creating User Interface to upload images
```
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('MyModel.h5')

# Create a function to preprocess the image
def preprocess_image(img):
    img = img.resize((512, 512))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

# Create a function to make predictions
def predict_category(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]
    labels = ['Normal', 'Stone', 'Tumor']
    category = labels[np.argmax(prediction)]
    return category

# Create the Tkinter GUI window
window = tk.Tk()
window.title("CT Scan Classifier")

# Create a function to handle image selection
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))
        image_tk = ImageTk.PhotoImage(image)
        image_label.configure(image=image_tk)
        image_label.image = image_tk  # Keep a reference to prevent garbage collection
        category = predict_category(image)
        result_label.configure(text="Predicted Category: " + category)

# Create the image label
image_label = tk.Label(window)
image_label.pack()

# Create the select image button
select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Create the result label
result_label = tk.Label(window, text="Predicted Category:")
result_label.pack()

# Start the Tkinter event loop
window.mainloop()
```
```
import os
import numpy as np
import matplotlib.pyplot as plt
Test_DIR = 'Testing Data/'
for i in os.listdir(Test_DIR):
    print(i)
    img1 = image.load_img(Test_DIR+i)
    plt.imshow(img1)
    plt.show()
```
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/b61860d2-93cf-4dd4-9f38-cc849300bb02)
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/81edbaec-9e79-4a02-a5ab-ec252ee2615c)
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/4e11e372-4f59-4866-9f6a-0fc36bfc4c73)
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/ac05d1da-3ed3-46ba-aeb0-5274d8e25499)

```
import os
import numpy as np
import matplotlib.pyplot as plt
Test_DIR = 'Testing Data/'
for i in os.listdir(Test_DIR):
    img1 = image.load_img(os.path.join(Test_DIR, i), target_size=(512,512))
    plt.imshow(img1)
    plt.show()
    X = image.img_to_array(img1)
    X = np.expand_dims(X,axis = 0)
    images = np.vstack([X])
    val = model.predict(images)
    predicted_class = np.argmax(val)
    print(i)
    if predicted_class == 0:
        print("Normal")
    elif predicted_class == 1:
        print("Stone")
    else:
        print("Tumor")
```
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/fadfe0dc-8325-4e84-8c72-a113e00b0340)
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/c5e1fafb-78ae-4ae4-87e5-b5386ea23752)
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/6a9f887d-5cd3-4aeb-a564-97cef76be0f9)
![image](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/39b9b11b-cc8d-4840-bcbf-bfa538ff97ad)

```
print(history.history)
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the graph
epochs = 25
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
![download](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/3cb6a821-6d77-44fb-90a7-599934e9653c)
![download (1)](https://github.com/geethikaxshoba/kidney-disease-detection/assets/97936145/8090cff8-43b9-4d01-b4a7-36021c282e59)


