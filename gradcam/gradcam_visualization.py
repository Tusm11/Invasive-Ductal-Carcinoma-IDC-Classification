import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

def get_img_array(img_path, size):
    # `img` is a PIL image of size 50x50
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (50, 50, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 50, 50, 3)
    array = np.expand_dims(array, axis=0)
    return array / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    return cam_path

if __name__ == "__main__":
    import glob
    import pickle

    models_to_test = ["vgg16", "resnet50"]
    base_path = 'data/breast_histopathology'
    
    # Find a positive (malignant) image for demonstration
    search_pattern = os.path.join(base_path, '*', '1', '*.png')
    positive_images = glob.glob(search_pattern, recursive=True)
    
    if not positive_images:
        print("No positive images found. Cannot run Grad-CAM demo.")
    else:
        sample_img = positive_images[0]
        print(f"Using sample image: {sample_img}")

        for model_type in models_to_test:
            model_path = f"models/{model_type}_model.pkl"
            if os.path.exists(model_path):
                print(f"\nProcessing {model_type}...")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Determine last conv layer
                if model_type == "vgg16":
                    last_conv_layer_name = "block5_conv3"
                else:
                    # For ResNet50, we need to find the correct layer name. 
                    # Keras ResNet50 usually has 'conv5_block3_out'
                    # Let's try to find it or use the last one containing 'conv' or 'out'
                    last_conv_layer_name = "conv5_block3_out"

                try:
                    img_array = get_img_array(sample_img, size=(50, 50))
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                    
                    output_dir = "gradcam/results"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        
                    cam_path = os.path.join(output_dir, f"{model_type}_gradcam.jpg")
                    save_and_display_gradcam(sample_img, heatmap, cam_path=cam_path)
                    print(f"Grad-CAM saved to {cam_path}")
                except Exception as e:
                    print(f"Error generating Grad-CAM for {model_type}: {e}")
            else:
                print(f"Model file {model_path} not found.")
