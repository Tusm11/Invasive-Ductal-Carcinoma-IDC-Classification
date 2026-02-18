import sys
import os

#Adding the project root to sys.path so 'models' folder is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.vgg16_classifier import build_vgg16
from models.resnet50_classifier import build_resnet50
import argparse

#Optimized for CPU training: Limit threads so multiple models can run 
#simultaneously without fighting for the same CPU resources.
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

def get_data_paths(base_path, sample_size=20000):
    """
    Scans the directory for image paths and labels.
    Sub-samples the data for efficient CPU training.
    """
    print("Enumerating images... this might take a moment.")
    all_images = glob.glob(os.path.join(base_path, '*', '*', '*.png'), recursive=True)
    
    if not all_images:
        print("No images found! Please ensure the dataset is downloaded and unzipped.")
        return None

    data = []
    for path in all_images:
        # Label is based on the folder name (0 or 1)
        # Path format: .../patient_id/[0 or 1]/image_name.png
        label = os.path.basename(os.path.dirname(path))
        data.append({'path': path, 'label': label})
    
    df = pd.DataFrame(data)
    #Stratified Sampling to maintain class ratio while reducing size for CPU
    if len(df) > sample_size:
        print(f"Sub-sampling dataset to {sample_size} images for CPU efficiency.")
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(int(sample_size/2), replace=False if len(x) > sample_size/2 else True))
    
    return df

def train(model_type='vgg16', epochs=10, batch_size=32, sample_size=10000):
    base_path = 'data/breast_histopathology'
    df = get_data_paths(base_path, sample_size=sample_size)
    
    if df is None: return

    #Split data
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

    #Image Generators (Reduced augmentation for CPU speed)
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='label',
        target_size=(50, 50), batch_size=batch_size, class_mode='binary'
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col='path', y_col='label',
        target_size=(50, 50), batch_size=batch_size, class_mode='binary'
    )

    #Build Model
    if model_type == 'vgg16':
        model = build_vgg16()
    else:
        model = build_resnet50()

    print(f"Starting training for {model_type} on CPU...")
    
    #Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f'models/{model_type}_best.h5', save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"Training complete. Model saved to models/{model_type}_best.h5")
    
    #Save as Pickle file as requested
    import pickle
    print(f"Saving {model_type} as pickle file...")
    with open(f'models/{model_type}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    #Also save training history as pickle for later plotting
    with open(f'models/{model_type}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
        
    print(f"Pickle files saved: models/{model_type}_model.pkl and models/{model_type}_history.pkl")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16', help='vgg16 or resnet50')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=10000)
    args = parser.parse_args()

    train(model_type=args.model, epochs=args.epochs, sample_size=args.sample_size)
