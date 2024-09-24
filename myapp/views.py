from django.shortcuts import render
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, LSTM, Dense, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Reshape
import io
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# import tensorflow_datasets as tfds
import math


train = pd.read_csv(
    'C:/Users/malir/OneDrive/Desktop/embryoclassification/dataset/hvwc23/train.csv')
test = pd.read_csv(
    'C:/Users/malir/OneDrive/Desktop/embryoclassification/dataset/hvwc23/test.csv')

FRAME_WIDTH = 255
FRAME_HEIGHT = 255
NUM_CLASSES = 2
BATCH_SIZE = 32
train, val = train_test_split(train, test_size=0.2, random_state=42)
def index(request):
    return render(request,'myapp/index.html')


def login(request):
    if request.method=="POST":
        username = request.POST['uname']
        password = request.POST['pwd']
        print(username,password)
        if username == 'admin' and password == 'admin123':
            return render(request, 'myapp/homepage.html')

    return render(request,'myapp/login.html')


def homepage(request):
    return render(request,'myapp/homepage.html')


def dataupload(request):

    # Assuming train.shape[0] and test.shape[0] hold the counts of training and test images respectively
    counts = [train.shape[0], test.shape[0]]
    labels = ['Training', 'Test']

    plt.figure(figsize=(6, 5))
    plt.bar(labels, counts)
    plt.ylabel('Number of Images')
    plt.title('Number of Images in Training and Test Sets')
    plt.show()

    print("Number of training images:", train.shape[0])
    print("Number of test images:", test.shape[0])
    content = {
        'train': train.shape[0],
        'test':test.shape[0],
    }

    return render(request,'myapp/dataupload.html',content)


def datapreprocessing(request):
    return render(request,'myapp/datapreprocessing.html')


def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(FRAME_HEIGHT, FRAME_WIDTH))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Define a custom generator
def custom_image_generator(dataframe, BATCH_SIZE, isTrain):
    num_samples = len(dataframe)

    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for i in range(0, num_samples, BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            batch_data = dataframe.iloc[batch_indices]
            batch_images = []
            batch_labels = batch_data['Class'].values.tolist()

            for _, row in batch_data.iterrows():
                if isTrain:
                    image_path = 'C:/Users/malir/OneDrive/Desktop/embryoclassification/dataset/hvwc23/train/'
                else:
                    image_path = 'C:/Users/malir/OneDrive/Desktop/embryoclassification/dataset/hvwc23/test/'
                image_path += row['Image']
                img = load_and_preprocess_image(image_path)
                batch_images.append(img)

            yield np.array(batch_images), np.array(batch_labels)
train_generator = custom_image_generator(train, BATCH_SIZE, True)
validation_generator = custom_image_generator(val, BATCH_SIZE,True)
validation_generator2 = custom_image_generator(val, BATCH_SIZE,True)

# Define a directory to save the model checkpoints
checkpoint_dir = '/model_checkpoints/'

# Ensure the directory exists
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_weights_vgg.h5'),  # Save with epoch number
    save_best_only=False,  # Save model weights after each epoch
    save_weights_only=True,  # Save only the weights, not the entire model
    verbose=1  # Display progress during saving
)

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if ((logs.get('accuracy')>=0.999)):
            print("\nLimits Reached cancelling training!")
            self.model.stop_training = True
end_callback = myCallback()

def createmodel(request):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False
    # Create a custom top classifier
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(
        Dense(NUM_CLASSES, activation='sigmoid'))  # Replace num_classes with the number of classes in your dataset

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Print the model summary to the redirected stdout
    model.summary()

    # Get the model summary as a string
    summary_string = sys.stdout.getvalue()

    # Reset stdout to its original value
    sys.stdout = original_stdout

    # Now, `summary_string` contains the model summary
    print(summary_string)
    content1={
        'data':summary_string
    }
    # Train the model using fit_generator
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=math.ceil(len(train) // BATCH_SIZE),  # Adjust this as needed
    #     epochs=7,  # Adjust the number of epochs as needed
    #     validation_data=validation_generator,
    #     validation_steps=math.ceil(len(val) // BATCH_SIZE),  # Adjust this as needed
    #     callbacks=[checkpoint_callback, end_callback])
    #
    # test_loss, test_accuracy = model.evaluate(validation_generator2, steps=math.ceil(len(val) // BATCH_SIZE))
    # print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    # file1 = open('accuracy.txt', 'w')
    # file1.write(f'Test accuracy: {test_accuracy * 100:.2f}%')
    # file1.close()
    # model.save('embryoclassification.h5')
    return render(request,'myapp/createmodel.html',content1)


def predict(imgname):
    train = pd.read_csv('C:/Users/malir/OneDrive/Desktop/embryoclassification/dataset/hvwc23/train.csv')
    test = pd.read_csv('C:/Users/malir/OneDrive/Desktop/embryoclassification/dataset/hvwc23/test.csv')

    # Assuming train.shape[0] and test.shape[0] hold the counts of training and test images respectively
    counts = [train.shape[0], test.shape[0]]
    labels = ['Training', 'Test']

    print("Number of training images:", train.shape[0])
    print("Number of test images:", test.shape[0])

    FRAME_WIDTH = 255
    FRAME_HEIGHT = 255

    # Define the function to load and preprocess the image
    def load_and_preprocess_image(image_path):
        img = load_img(image_path, target_size=(FRAME_HEIGHT, FRAME_WIDTH))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array

    Pred_model = load_model('C:/Users/malir/OneDrive/Desktop/embryoclassification/model_checkpoints/embryoclassification.h5')

    print("Model Summary:")
    print(Pred_model.summary())

    # Define the path to your image file
    image_path = imgname

    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)

    # Make predictions
    predictions = Pred_model.predict(np.array([img]))

    # Assuming it's a classification task, get the predicted class
    predicted_class = np.argmax(predictions, axis=1)

    if predicted_class == 0:
        embryo_label = 'Good Stage : A good quality embryo typically exhibits well-formed, evenly sized cells without severe fragmentation.\n These embryos are more likely to successfully implant and develop into a viable pregnancy,\n boasting higher chances of contributing to a successful conception'
    else:
        embryo_label = 'Bad stage : This embryo is severely fragmented and has unevenly sized cells. \n It is a poor quality embryo.\n This embryo probably does not have much chance to implant and make a viable pregnancy'

    print(f"The predicted label for the embryo is in {embryo_label}")
    return embryo_label


def prediction(request):

    global result
    print('hi-------------')
    if request.method=='POST':
        imgname = request.POST['myFile']
        imgpath='C:/Users/malir/OneDrive/Desktop/embryoclassification/testimages/'
        print(imgname)
        imgafile=imgpath+imgname
        res=predict(imgafile)

        content={
            'data':res,
            'imgs':imgname
        }
        return render(request, 'myapp/prediction.html',content)

    return render(request,'myapp/prediction.html')


def viewgraph(request):
    return render(request,'myapp/viewgraph.html')

