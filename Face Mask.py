#!/usr/bin/env python
# coding: utf-8

# In[60]:


# Common Python libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# For reading in images and image manipulation
import cv2
import tensorflow as tf
# For label encoding the target variable
from sklearn.preprocessing import LabelEncoder

# For tensor based operations
from tensorflow.keras.utils import to_categorical, normalize

# For Machine Learning
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# For face detection
from mtcnn.mtcnn import MTCNN
import flask


# In[61]:


# Reading in the csv file
train = pd.read_csv( r"C:\archive\train.csv")

# Displaying the first five rows
train.head()


# In[62]:


# Total number of unique images
len(train["name"].unique())


# In[63]:


# classnames to select
options = ["face_with_mask", "face_no_mask"]

# Select rows that have the classname as either "face_with_mask" or "face_no_mask"
train = train[train["classname"].isin(options)].reset_index(drop=True)
train.sort_values("name", axis=0, inplace=True)


# In[64]:


#Plotting a bar plot
x_axis_val = ["face_with_mask", "face_no_mask"]
y_axis_val = train.classname.value_counts()
plt.bar(x_axis_val, y_axis_val)


# In[65]:


# Contains images of medical masks
images_file_path = "C:/archive/Medical mask/Medical mask/Medical Mask/images/"
# Fetching all the file names in the image directory
image_filenames = os.listdir(images_file_path)

# Printing out the first five image names
print(image_filenames[:5])


# In[66]:


# Getting the full image filepath
sample_image_name = train.iloc[0]["name"]
sample_image_file_path = images_file_path + sample_image_name

# Select rows with the same image name as in the "name" column of the train dataframe
sel_df = train[train["name"] == sample_image_name]

# Convert all of the available "bbox" values into a list
bboxes = sel_df[["x1", "x2", "y1", "y2"]].values.tolist()

# Creating a figure and a sub-plot
fig, ax = plt.subplots()

# Reading in the image as an array
img = plt.imread(sample_image_file_path)

# Showing the image
ax.imshow(img)

# Plotting the bounding boxes
for box in bboxes:

    x1, x2, y1, y2 = box

    # x and y co-ordinates
    xy = (x1, x2)

    # Width of box
    width = y1 - x1

    # Height of box
    height = y2 - x2

    rect = patches.Rectangle(
        xy,
        width,
        height,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )

    ax.add_patch(rect)


# In[67]:


img_size = 50
data = []
for index, row in train.iterrows():
 # Single row
 name, x1, x2, y1, y2, classname = row.values
 # Full file path
 full_file_path = images_file_path + name
 # Reading in the image array as a grayscale image
 img_array = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)
 # Selecting the portion covered by the bounding box
 crop_image = img_array[x2:y2, x1:y1]
 # Resizing the image
 new_img_array = cv2.resize(crop_image, (img_size, img_size))
 # Appending the arrays into a data variable along with bounding box
 data.append([new_img_array, classname])
# Plotting one of the images after pre-processing
plt.imshow(data[0][0], cmap="gray")


# In[68]:


# Initializing an empty list for features (independent variables)
x = []
# Initializing an empty list for labels (dependent variable)
y = []
for features, labels in data:
 x.append(features)
 y.append(labels)


# In[69]:


# Reshaping the feature array (Number of images, IMG_SIZE, IMG_SIZE, Color depth)
x = np.array(x).reshape(-1, 50, 50, 1)
# Normalizing
x = normalize(x, axis=1)
# Label encoding y
lbl = LabelEncoder()
y = lbl.fit_transform(y)
# Converting it into a categorical variable
y = to_categorical(y)
input_img_shape = x.shape[1:]
print(input_img_shape)


# In[70]:


#Initializing a sequential keras model
model = Sequential()
# Adding a 2D convolution layer
model.add(
 Conv2D(
 filters=100,
 kernel_size=(3, 3),
 use_bias=True,
 input_shape=input_img_shape,
 activation="relu",
 strides=2,
 )
)
# Adding a max-pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a 2D convolution layer - Output Shape = 10 x 10 x 64
model.add(Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, activation="relu"))
# Adding a max-pooling layer - Output Shape = 5 x 5 x 64
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a flatten layer - Output Shape = 5 x 5 x 64 = 1600
model.add(Flatten())
# Adding a dense layer - Output Shape = 50
model.add(Dense(50, activation="relu"))
# Adding a dropout
model.add(Dropout(0.2))
# Adding a dense layer with softmax activation
model.add(Dense(2, activation="softmax"))
# Printing the model summary
model.summary()


# In[71]:


opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay=1e-5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x,y,epochs=30,batch_size=5)


# In[79]:


# Image file path for sample image
test_image_file_path =r"C:\archive\Medical mask\Medical mask\Medical Mask\images\0286.jpg"
# Loading in the image
img = plt.imread(test_image_file_path)
# Showing the image
plt.imshow(img)

    


# In[80]:


# Initializing the detector
detector = MTCNN()
# Detecting the faces in the image
faces = detector.detect_faces(img)
print(faces)


# In[81]:


# Reading in the image as a grayscale image
img_array = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
# Initializing the detector
detector = MTCNN()
# Detecting the faces in the image
faces = detector.detect_faces(img)
# Getting the values for bounding box
x1, x2, width, height = faces[0]["box"]
# Selecting the portion covered by the bounding box
crop_image = img_array[x2 : x2 + height, x1 : x1 + width]
# Resizing the image
new_img_array = cv2.resize(crop_image, (img_size, img_size))
# Plotting the image
plt.imshow(new_img_array, cmap="gray")


# In[82]:


# Reshaping the image
x = new_img_array.reshape(-1, 50, 50, 1)
# Normalizing
x = normalize(x, axis=1)


# In[83]:


prediction = model.predict(x)
print(prediction)


# In[94]:


# Returns the index of the maximum value
np.argmax(prediction)


# In[95]:


if np.argmax(prediction)==0:
    print("NO face mask....please wear face Mask")
else:
    print("Weared Face Mask")


# In[ ]:




