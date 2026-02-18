import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
from sklearn.model_selection import train_test_split
import pickle

def evaluate(model_path, model_name):
    print(f"Loading model from {model_path}...")
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = tf.keras.models.load_model(model_path)
    
    #needed to reconstruct the test set (same logic as training)
    base_path = 'data/breast_histopathology'
    all_images = glob.glob(os.path.join(base_path, '*', '*', '*.png'), recursive=True)
    data = []
    for path in all_images[:20000]: # Using same sample limit for speed
        label = os.path.basename(os.path.dirname(path))
        data.append({'path': path, 'label': label})
    
    df = pd.DataFrame(data)
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col='path', y_col='label',
        target_size=(50, 50), batch_size=32, class_mode='binary',
        shuffle=False
    )

    print("Running predictions...")
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_gen.classes

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

    # AUC Score
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'evaluation/cm_{model_name}.png')
    print(f"Confusion matrix saved to evaluation/cm_{model_name}.png")

if __name__ == "__main__":
    # Prioritize pickle files for evaluation if they exist
    if os.path.exists('models/vgg16_model.pkl'):
        evaluate('models/vgg16_model.pkl', 'VGG16')
    elif os.path.exists('models/vgg16_best.h5'):
        evaluate('models/vgg16_best.h5', 'VGG16')
    
    if os.path.exists('models/resnet50_model.pkl'):
        evaluate('models/resnet50_model.pkl', 'ResNet50')
    elif os.path.exists('models/resnet50_best.h5'):
        evaluate('models/resnet50_best.h5', 'ResNet50')
