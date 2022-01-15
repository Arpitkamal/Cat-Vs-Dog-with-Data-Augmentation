from cmath import log
from gc import callbacks
import streamlit as st
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, Adadelta, SGD, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from io import StringIO
import plotly.express as px 
import plotly.graph_objs as go
import pandas as pd


st.header("Cat Vs Dog ")

selectedway = st.selectbox("Cat Vs Dog",("Without Data Augumentation","With Data Augumenation"))

@st.cache
def get_data():
    pwd = os.getcwd()
    # st.write(pwd)
    base_dir = os.path.join(pwd,'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')

    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')

    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    return train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_dir, validation_dir



if selectedway == "Without Data Augumentation":
    # Load the data 
    def without_aug():
        
        train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_dir, validation_dir = get_data()

        st.write("Total Training Cat Images :",len(os.listdir(train_cats_dir ) ))
        st.write("Total Training Dogs Images :",len(os.listdir(train_dogs_dir ) ))
        st.write("Total Validation Cat Images :",len(os.listdir(validation_cats_dir ) ))
        st.write("Total Validation Cat Images",len(os.listdir(validation_dogs_dir ) ))

        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                st.write("Epoch",epoch,logs)

        callbacks = myCallback()

        st.subheader("**Build Model**")

        act_fun = st.selectbox("Select the activation function for Convolution Neural networks:",('relu','sigmoid', 'tanh'),index=0)

        pad_conv = st.selectbox("Select the Padding for Convolution Neural networks:",('same','valid'),index=0)

        Conv_filter1 = st.slider("Select the Filter size for the First Convolution Layer",32,128,step=32,value=32)
        Conv_filter2 = st.slider("Select the Filter size for the Second Convolution Layer",32,128,step=32,value=64)
        Conv_filter3 = st.slider("Select the Filter Size for the Third Convolution Layer",32,128,step=32,value=128)
        Conv_filter4 = st.slider("Select the Filter Size for the Fourth Convolution Layer",32,128,step=32,value=128)

        act_dense = st.selectbox("Select the activation function for the Dense Layer",('relu','sigmoid', 'tanh'),index=0)

        epochs_model = st.slider("Select the Number of Epochs :", 0,100,step=10,value=1)

        loss_func = st.selectbox("Select the loss Function for model Compiler: ",('binary_crossentropy',                                                                                
                                                                                'categorical_crossentropy',
                                                                                'mean_absolute_error',
                                                                                'mean_absolute_percentage_error',
                                                                                'mean_squared_error',
                                                                                'sparse_categorical_crossentropy'),index=0)

        optimizer_com = st.selectbox("Select the optimization Function: ",('RMSprop','Adam','SGD','Nadam','Adadelta'),index=0)

        st.subheader("** Selected Hyperparameters**")
        st.write(f"Activation Function used in Convolution Neural Network is: `{act_fun}`", )
        st.write(f"Padding used in Convolutional Neural Network is: `{pad_conv}`" )
        st.write(f"Filter size used in First Convolution layer is : `{Conv_filter1}`")
        st.write(f"Filter size used in Second Convolution layer is : `{Conv_filter2}`")
        st.write(f"Filter size used in Third Convolution layer is : `{Conv_filter3}`")
        st.write(f"Filter size used in Fourth Convolution layer is : `{Conv_filter4}`")
        st.write(f"Activation Function used in Dense Layer is : `{act_dense}`")
        st.write(f"Number of epochs used is : `{epochs_model}`" )
        st.write(f"Loss function used in the compiler is : `{loss_func}`")
        st.write(f"Optimization function used in the compiler is: `{optimizer_com}`")


        # for printing model summary on the application
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        if st.button("Train Model"):
            st.balloons()
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters = Conv_filter1, kernel_size= (3,3),activation=act_fun,padding=pad_conv, input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(filters= Conv_filter2, kernel_size=(3,3), activation=act_fun,padding=pad_conv),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters=Conv_filter3, kernel_size=(3,3), activation=act_fun,padding=pad_conv),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters= Conv_filter4, kernel_size=(3,3), activation=act_fun,padding=pad_conv),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=act_dense),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            st.subheader("**Model Summary**")
            model.summary()

            sys.stdout = old_stdout
            st.text(mystdout.getvalue())

            if optimizer_com == 'RMSprop':
                model.compile(loss=loss_func,optimizer=RMSprop(learning_rate=0.001),
                metrics=['accuracy'])
            elif optimizer_com == 'Adam':
                model.compile(loss=loss_func,optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
            elif optimizer_com == 'SGD':
                model.compile(loss=loss_func,optimizer=SGD(learning_rate=0.001),metrics=['accuracy'])
            elif optimizer_com == 'Nadam':
                model.compile(loss=loss_func,optimizer=Nadam(learning_rate=0.001),metrics=['accuracy'])
            else:
                model.compile(loss=loss_func,optimizer=Adadelta(learning_rate=0.001),metrics=['accuracy'])


            # All images will be rescaled by 1./255 (Normalizing the data)
            train_datagen = ImageDataGenerator(rescale=1./255)
            test_datagen = ImageDataGenerator(rescale=1./255)

            # Flow training images in batches of 20 using train_datagen generator
            train_generator = train_datagen.flow_from_directory(
                    train_dir,  # This is the source directory for training images
                    target_size=(150, 150),  # All images will be resized to 150x150
                    batch_size=20,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary')

            # Flow validation images in batches of 20 using test_datagen generator
            validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=20,
                    class_mode='binary')
            
            st.subheader("Model Training \n")

            with st.spinner("**Training the model may take time, so grab a Coffe and Enjoy**"):
                history = model.fit(
                    train_generator,
                    steps_per_epoch=100,  # 2000 images = batch_size * steps
                    epochs=epochs_model,
                    validation_data=validation_generator,
                    validation_steps=50,  # 1000 images = batch_size * steps
                    verbose=2,
                    callbacks=[callbacks] 
                    )
            st.success("Model Train is Completed")
            st.balloons()
           

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']


            epochs = list(range(len(acc)))

            st.subheader("**Training and validation Accuracy**")

            # Ploting acucuracy of the model on training and validation data
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=epochs, y=acc,
                                mode='lines',
                                name='Training Accuracy',
                                ))
            fig1.add_trace(go.Scatter(x=epochs, y=val_acc,
                                mode='lines+markers',
                                name='Validation Accuracy'))

            fig1.update_layout(
                        xaxis_title='Epochs',
                        yaxis_title='Accuracy')

            st.plotly_chart(fig1)

            st.subheader("**Training and validation Loss**")

            # Ploting loss of the model on training and validation data
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=epochs, y=loss,
                                mode='lines',
                                name='Training Loss',
                                ))
            fig2.add_trace(go.Scatter(x=epochs, y=val_loss,
                                mode='lines+markers',
                                name='Validation Loss'))

            fig2.update_layout(#title='Training and validation Loss',
                        xaxis_title='Epochs',
                        yaxis_title='Loss')

            st.plotly_chart(fig2)
    # calling without Augumentaion function
else:
    st.write("Cat Vs Dog With Data Augumentation")

    def with_aug():
        train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, train_dir, validation_dir = get_data()

        st.write("Total Training Cat Images :",len(os.listdir(train_cats_dir ) ))
        st.write("Total Training Dogs Images :",len(os.listdir(train_dogs_dir ) ))
        st.write("Total Validation Cat Images :",len(os.listdir(validation_cats_dir ) ))
        st.write("Total Validation Cat Images",len(os.listdir(validation_dogs_dir ) ))

        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                st.write("Epoch",epoch,logs)

        callbacks = myCallback()

        st.subheader("**Buid Model**")

        act_fun = st.selectbox("Select the activation function for Convolution Neural networks:",('relu','sigmoid', 'tanh'),index=0)

        pad_conv = st.selectbox("Select the Padding for Convolution Neural networks:",('same','valid'),index=0)

        Conv_filter1 = st.slider("Select the Filter size for the First Convolution Layer",32,128,step=32,value=32)
        Conv_filter2 = st.slider("Select the Filter size for the Second Convolution Layer",32,128,step=32,value=64)
        Conv_filter3 = st.slider("Select the Filter Size for the Third Convolution Layer",32,128,step=32,value=128)
        Conv_filter4 = st.slider("Select the Filter Size for the Fourth Convolution Layer",32,128,step=32,value=128)

        act_dense = st.selectbox("Select the activation function for the Dense Layer",('relu','sigmoid', 'tanh'),index=0)

        epochs_model = st.slider("Select the Number of Epochs :", 0,100,step=10,value=1)

        loss_func = st.selectbox("Select the loss Function for model Compiler: ",('binary_crossentropy',                                                                                
                                                                                'categorical_crossentropy',
                                                                                'mean_absolute_error',
                                                                                'mean_absolute_percentage_error',
                                                                                'mean_squared_error',
                                                                                'sparse_categorical_crossentropy'),index=0)

        optimizer_com = st.selectbox("Select the optimization Function: ",('RMSprop','Adam','SGD','Nadam','Adadelta'),index=0)

        st.subheader("** Selected Hyperparameters**")
        st.write(f"Activation Function used in Convolution Neural Network is: `{act_fun}`", )
        st.write(f"Padding used in Convolutional Neural Network is: `{pad_conv}`" )
        st.write(f"Filter size used in First Convolution layer is : `{Conv_filter1}`")
        st.write(f"Filter size used in Second Convolution layer is : `{Conv_filter2}`")
        st.write(f"Filter size used in Third Convolution layer is : `{Conv_filter3}`")
        st.write(f"Filter size used in Fourth Convolution layer is : `{Conv_filter4}`")
        st.write(f"Activation Function used in Dense Layer is : `{act_dense}`")
        st.write(f"Number of epochs used is : `{epochs_model}`" )
        st.write(f"Loss function used in the compiler is : `{loss_func}`")
        st.write(f"Optimization function used in the compiler is: `{optimizer_com}`")

        st.subheader("**Data Augmentation on Training and Testing data**")

        rot_range = st.slider("Select the Rotation range : ",0,120,step=10,value=40)
        width_range = st.slider("Select the width shift range : ",0.0,1.0,step=0.1,value=0.2)
        height_range = st.slider("Select the height shift range :",0.0,1.0,step=0.1,value=0.2)
        shear_range = st.slider("Select the Shear range :",0.0,1.0,step=0.1,value=0.2)
        zoom_range = st.slider("Select the Zoom range :",0.0,1.0,step=0.1,value=0.2)
        hor_flip = st.selectbox("Select the Horizontel Flip :",(True,False),index=0)
        fill_mode = st.selectbox("Select the File mode : ",("constant","nearest","reflect","wrap"),index=1)

        # for printing model summary on the application
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        if st.button("Train Model"):
            st.balloons()
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters = Conv_filter1, kernel_size= (3,3),activation=act_fun,padding=pad_conv, input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(filters= Conv_filter2, kernel_size=(3,3), activation=act_fun,padding=pad_conv),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters=Conv_filter3, kernel_size=(3,3), activation=act_fun,padding=pad_conv),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters= Conv_filter4, kernel_size=(3,3), activation=act_fun,padding=pad_conv),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=act_dense),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            st.subheader("**Model Summary**")
            model.summary()

            sys.stdout = old_stdout
            st.text(mystdout.getvalue())

            if optimizer_com == 'RMSprop':
                model.compile(loss=loss_func,optimizer=RMSprop(learning_rate=0.001),
                metrics=['accuracy'])
            elif optimizer_com == 'Adam':
                model.compile(loss=loss_func,optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
            elif optimizer_com == 'SGD':
                model.compile(loss=loss_func,optimizer=SGD(learning_rate=0.001),metrics=['accuracy'])
            elif optimizer_com == 'Nadam':
                model.compile(loss=loss_func,optimizer=Nadam(learning_rate=0.001),metrics=['accuracy'])
            else:
                model.compile(loss=loss_func,optimizer=Adadelta(learning_rate=0.001),metrics=['accuracy'])

            
            

            
            # This code has changed. Now instead of the ImageGenerator just rescaling
            # the image, we also rotate and do other operations
            # Updated to do image augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=rot_range,
                width_shift_range=width_range,
                height_shift_range=height_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=hor_flip,
                fill_mode=fill_mode)
            
            test_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=rot_range,
                width_shift_range=width_range,
                height_shift_range=height_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                horizontal_flip=hor_flip,
                fill_mode=fill_mode
                )

            # Flow training images in batches of 20 using train_datagen generator
            train_generator = train_datagen.flow_from_directory(
                    train_dir,  # This is the source directory for training images
                    target_size=(150, 150),  # All images will be resized to 150x150
                    batch_size=20,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary')

            # Flow validation images in batches of 20 using test_datagen generator
            validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=20,
                    class_mode='binary')

            with st.spinner("**Training the model may take time, so grab a Coffe and Enjoy**"):
                history = model.fit(
                train_generator,
                steps_per_epoch=100,  # 2000 images = batch_size * steps
                epochs=epochs_model,
                validation_data=validation_generator,
                validation_steps=50,  # 1000 images = batch_size * steps
                verbose=2,
                callbacks=[callbacks])
            
            st.success("Model Train is Completed")
            st.balloons()


            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']


            epochs = list(range(len(acc)))

            # Ploting acucuracy of the model on training and validation data
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=epochs, y=acc,
                                mode='lines',
                                name='Training Accuracy',
                                ))
            fig1.add_trace(go.Scatter(x=epochs, y=val_acc,
                                mode='lines+markers',
                                name='Validation Accuracy'))

            fig1.update_layout(title='Training and validation Accuracy',
                        xaxis_title='Epochs',
                        yaxis_title='Accuracy')

            st.plotly_chart(fig1)

            # Ploting loss of the model on training and validation data
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=epochs, y=loss,
                                mode='lines',
                                name='Training Loss',
                                ))
            fig2.add_trace(go.Scatter(x=epochs, y=val_loss,
                                mode='lines+markers',
                                name='Validation Loss'))

            fig2.update_layout(title='Training and validation Loss',
                        xaxis_title='Epochs',
                        yaxis_title='Loss')

            st.plotly_chart(fig2)


if selectedway == "Without Data Augumentation":
    without_aug()
else:
    with_aug()


                



    

        
        

        

        







