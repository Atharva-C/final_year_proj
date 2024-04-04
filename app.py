import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the model with custom objects
def IoU(y_val, y_pred):
    class_iou = []
    n_classes = 7

    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)

    for c in range(n_classes):
        TP = np.sum((y_truei == c) & (y_predi == c))
        FP = np.sum((y_truei != c) & (y_predi == c))
        FN = np.sum((y_truei == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        if(float(TP + FP + FN) == 0):
          IoU=TP/0.001
        class_iou.append(IoU)
    MIoU=sum(class_iou)/n_classes
    return MIoU

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

with tf.keras.utils.custom_object_scope({'miou': miou, 'specificity': specificity, 'sensitivity': sensitivity, 'f1_score_metric': f1_score_metric}):
    fused_model = tf.keras.models.load_model('FUSED_model.h5')

st.title('Image Semantic Segmentation')

uploaded_image1 = st.file_uploader('Upload the first image', type=['jpg', 'png'])
uploaded_image2 = st.file_uploader('Upload the second image', type=['jpg', 'png'])
predict_button = st.button('Predict')

if uploaded_image1 is not None and uploaded_image2 is not None and predict_button:
    # Read the images
    image1 = Image.open(uploaded_image1)
    image2 = Image.open(uploaded_image2)
    st.image([image1, image2], caption=['Uploaded Image 1', 'Uploaded Image 2'], use_column_width=True)

    # Preprocess the images
    input_image1 = image1.resize((320, 256))
    input_image1 = np.array(input_image1) / 255.0
    input_image1 = np.expand_dims(input_image1, axis=0)

    input_image2 = image2.resize((320, 256))
    input_image2 = np.array(input_image2) / 255.0
    input_image2 = np.expand_dims(input_image2, axis=0)

    # Perform segmentation
    segmentation_mask = fused_model.predict([input_image1, input_image2])

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
