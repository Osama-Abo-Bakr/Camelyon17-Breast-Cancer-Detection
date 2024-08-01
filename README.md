# Camelyon17 Breast Cancer Detection

## Overview

This project aims to develop a breast cancer detection model using the Camelyon17 dataset. The dataset comprises histopathological images of breast cancer tissues, which we use to classify the images into cancerous and non-cancerous categories. The project involves data exploration, model design, training, evaluation, and deployment.

## 1. Prerequisites

To run this code, ensure that the following libraries are installed:

```bash
pip install -U efficientnet tensorflow numpy matplotlib pandas pillow scikit-learn
```

## 2. Data Loading and Exploration

The dataset used is the [Camelyon17](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview) dataset. We start by loading and visualizing the data to understand its structure and the distribution of classes.

### Code

```python
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import * 
import os
import shutil
import json
from PIL import Image
```

**Data Loading Function**

```python
def train_img_path(id_str):
    return os.path.join(r"/kaggle/input/histopathologic-cancer-detection/train", f"{id_str}.tif")
```

### Exploratory Data Analysis

We explore a sample image and display a few examples of both cancerous and non-cancerous images.

```python
example_path = "/kaggle/input/histopathologic-cancer-detection/train/f38a6374c348f90b587e046aac6079959adf3835.tif"
example_img = Image.open(example_path)
example_array = np.array(example_img)
print(f"Image Shape = {example_array.shape}")
plt.imshow(example_img)
plt.show()

train_labels_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_labels_df["filename"] = train_labels_df["id"].apply(train_img_path)
train_labels_df["label"] = train_labels_df["label"].astype(str)
train_labels_df.head()

train_labels_df.shape
set(train_labels_df['label'])
```

## 3. Model Design

We designed a custom ResNet50 model, leveraging the strengths of residual networks to address the image classification task. We also experimented with EfficientNetB0 and Vision Transformers (ViT) to compare performance.

### Custom ResNet50 Implementation

```python
def identity_block(X, f, filters, training=True, initializer=random_uniform):
    # Implementation details...
    return X

def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
    # Implementation details...
    return X

def ResNet50(input_shape=(96, 96, 3)):
    # Implementation details...
    return model
```

### EfficientNetB0 Implementation

```python
import efficientnet.tfkeras as efn

base_model = efn.EfficientNetB0(input_shape=(96,96,3), include_top=False, weights='imagenet')
cnn = Sequential([
    base_model,
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
```

### Vision Transformer (ViT) Implementation

```python
from transformers import ViTForImageClassification, ViTokenizer
# Implementation details...
```

## 4. Model Training and Evaluation

### Training

We train the models using the Adam optimizer and binary cross-entropy loss.

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.AUC()])

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=val_steps,
    epochs=10
)
```

### Evaluation

Evaluation metrics include accuracy, precision, recall, F1 score, and ROC-AUC. Since the test labels are not available, evaluation is conducted on the validation set.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

val_predictions = tf.nn.sigmoid(model.predict(validation_generator)).numpy()
val_pred_classes = (val_predictions > 0.5).astype(int).flatten()
true_labels = validation_generator.classes

accuracy = accuracy_score(true_labels, val_pred_classes)
precision = precision_score(true_labels, val_pred_classes)
recall = recall_score(true_labels, val_pred_classes)
f1 = f1_score(true_labels, val_pred_classes)
roc_auc = roc_auc_score(true_labels, val_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
```

### Results

Below are the training and validation metrics for ResNet and EfficientNetB0.

#### ResNet Metrics

![ResNet Metrics](download.png)

#### EfficientNetB0 Metrics

![EfficientNetB0 Metrics](0.png)

## 5. References

1. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [Link](https://arxiv.org/abs/1512.03385)
2. **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [Link](https://arxiv.org/abs/1905.11946)
3. **Vision Transformer**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., & others (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *In Proceedings of the International Conference on Learning Representations (ICLR)*. [Link](https://arxiv.org/abs/2010.11929)
