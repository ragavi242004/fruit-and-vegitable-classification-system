import streamlit as st
import tensorflow as tf
import numpy as np

def model_Prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5") 
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img1.webp"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an image:")
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", width=300, use_column_width=True)
        if st.button("Predict"):
            st.balloons()
            prediction = model_Prediction(test_image)
            result_index=model_Prediction(test_image)
            # Reading Labels
            with open("label.txt") as f:
                content = f.readlines()
                labels = [x.strip() for x in content]  # Remove whitespace characters
            st.write(f"Predicted Label: {labels[prediction]}")
            st.success("Model is Predicting it's a {}".format(labels[result_index]))
