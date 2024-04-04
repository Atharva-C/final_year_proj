import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import backend as K

import numpy as np

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# Load the model with custom objects
from keras import backend as K

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def miou( y_true, y_pred ) :
    score = tf.py_function( lambda y_true, y_pred : IoU( y_true, y_pred).astype('float32'),
                        [y_true, y_pred],
                        'float32')
    return score

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
def f1_score_metric(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='float32')
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

class global_flow(tf.keras.Model):
    def __init__(self,name="global_flow"):
        super().__init__(name=name)
        self.globalavglayer = GlobalAveragePooling2D()
        self.BN = BatchNormalization(axis=3)
        self.activation = Activation('relu')


    def build(self, input_shape):

        self.conv = Conv2D(input_shape[-1],(1,1),padding='same'
                           )
        self.upsample = UpSampling2D((input_shape[1],input_shape[2]),interpolation='bilinear')

    def call(self, X):
        X = self.globalavglayer(X)
        X = tf.expand_dims(X,axis=1)
        X = tf.expand_dims(X,axis=1)
        X = self.BN(X)
        X = self.activation(X)
        X = self.conv(X)
        X = self.upsample(X)

        return X

class context_flow(tf.keras.Model):
    def __init__(self,N=2):
        super().__init__()
        self.concat = Concatenate()
        self.mult = Multiply()
        self.add = Add()
        self.avgpool = AveragePooling2D(pool_size=(2,2),strides=N)

        self.reluact = Activation('relu')

        self.sigmoidact = Activation(sigmoid)
        self.upconv = UpSampling2D((N,N),interpolation='bilinear')


    def build(self,X):
        INP,FLOW = X[0], X[1]
        global_output_shape = FLOW

        self.conv1 = Conv2D(global_output_shape[-1],(3,3),padding='same')
        self.conv2 = Conv2D(global_output_shape[-1],(3,3),padding='same')

        self.conv3 = Conv2D(global_output_shape[-1],(1,1),padding='same')
        self.conv4 = Conv2D(global_output_shape[-1],(1,1),padding='same')

    def call(self, X):
        INP, FLOW = X[0], X[1]
        X = self.concat([INP,FLOW])
        X = self.avgpool(X)
        X = self.conv1(X)
        Y = self.conv2(X)
        X = self.conv3(Y)
        X = self.reluact(X)
        X = self.conv4(X)
        X = self.sigmoidact(X)
        X = self.mult([Y,X])
        X = self.add([Y,X])
        X = self.upconv(X)

        return X

class fsm(tf.keras.layers.Layer):
    def __init__(self, name="feature_selection"):
        super().__init__(name=name)
        self.conv_1 = Conv2D(32 ,kernel_size=(3,3),padding='same')
        self.global_avg_pool = GlobalAveragePooling2D()
        self.conv_2 = Conv2D(32 ,kernel_size=(1,1),padding='same')
        self.bn =BatchNormalization()
        self.act_sigmoid= Activation('sigmoid')
        self.multiply =Multiply()
        self.upsample = UpSampling2D(size=(2,2),interpolation='bilinear')

    def call(self, X):
        X= self.conv_1(X)
        global_avg = self.global_avg_pool(X)
        global_avg= tf.expand_dims(global_avg, 1)
        global_avg = tf.expand_dims(global_avg, 1)
        conv1= self.conv_2(global_avg)
        bn1= self.bn(conv1)
        Y = self.act_sigmoid(bn1)
        output = self.multiply([X, Y])
        FSM_Conv_T = self.upsample(output)

        return FSM_Conv_T

class agcn(tf.keras.layers.Layer):
    def __init__(self, name="global_conv_net"):
        super().__init__(name=name)
        self.conv_1  = Conv2D(32,kernel_size=(1,7),padding='same')
        self.conv_2  = Conv2D(32,kernel_size=(7,1),padding='same')
        self.conv_3  = Conv2D(32,kernel_size=(1,7),padding='same')
        self.conv_4  = Conv2D(32,kernel_size=(7,1),padding='same')
        self.conv_3  = Conv2D(32,kernel_size=(3,3),padding='same')
        self.add = Add()

    def call(self, X):
        conv1 = self.conv_1(X)
        conv2= self.conv_2(conv1)
        conv3 = self.conv_4(X)
        conv4 = self.conv_3(conv3)
        add1 = self.add([conv2,conv4])
        conv5 = self.conv_3(add1)
        X = self.add([conv5,add1])

        return X

with tf.keras.utils.custom_object_scope({'iou_coef': iou_coef}, {'dice_coef': dice_coef}, {'specificity': specificity},
                                        {'sensitivity': sensitivity}, {'f1_score_metric': f1_score_metric},
                                        {'global_flow': global_flow}, {'context_flow': context_flow}, {'fsm': fsm},
                                        {'agcn': agcn}):
    CANet_15epoch = tf.keras.models.load_model('CANet_50epoch.h5')
print("Model loaded successfully.")

st.title('Autonomous Vehicle Scene Understanding: Image Segmentation')

uploaded_image = st.file_uploader('Upload the image', type=['jpg', 'png'])
predict_button = st.button('Predict')


if uploaded_image is not None and predict_button:
    # Read the image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    input_image = image.resize((256, 256))
    input_image = np.array(input_image) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Perform segmentation
    segmentation_mask = CANet_15epoch.predict(input_image)

    # Post-process the segmentation mask
    segmentation_mask = np.argmax(segmentation_mask, axis=-1)
    segmentation_mask = np.squeeze(segmentation_mask, axis=0)
    segmentation_mask = np.array(segmentation_mask, dtype=np.uint8)

    # Display the segmented image
    fig, ax = plt.subplots(figsize=(320/100, 256/100))
    ax.imshow(segmentation_mask, cmap='jet')
    ax.axis('off')
    ax.margins(0, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig)

