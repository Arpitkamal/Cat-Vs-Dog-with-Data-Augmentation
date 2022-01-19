import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time

fig = plt.figure()

with open("app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Cat Vs Dog With Data Augmentation")

st.write("You can upload the image to check whether It is Cat or Dog using Pre-trained model")

model_selected = st.selectbox("Select the model",("Model With Data Augmentation","Model With Transfer Learning and Data Augmentation"),index=1)
st.write(f'Selected Model is : `{model_selected}`')

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.balloons()
                if predictions > 0.5:
                    st.write("Dog")
                else:
                    st.write("Cat")
                    


def predict(image):
    st.write(model_selected)
    if model_selected == "Model With Data Augmentation":
        classifier_model = "Cat_dog_model.h5"
    else:
        classifier_model = "Cat_dog_transfer.h5" 
    IMAGE_SHAPE = (255, 255,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((150,150))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    classes = model.predict(test_image, batch_size=10)
    return classes

main()
