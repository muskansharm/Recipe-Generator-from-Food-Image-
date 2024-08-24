import streamlit as st
from PIL import Image
import tensorflow as tf
import google.generativeai as genai


# run cmd as administrator
# install python
# pip install streamlit pillow numpy tensorflow google-generativeai
# Setup Google cloud key https://ai.google.dev/tutorials/setup
# replace key for ???
# download pycharm and open this file in pycharm
# Get streamlit link and run in cmd
# download food image from google and run the web page


# Set up Google Cloud credentials
genai.configure(api_key='AIzaSyAZx24JuA9IVc-6T5nYzDVo00NhqMT-va8')

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)

# Function to predict food name
def predict_food_name(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0]
    food_name = decoded_preds[0][1]
    return food_name

# Function to generate recipe using Google Cloud TTS
def generate_recipe(food_name):
    model = genai.GenerativeModel('gemini-pro')
    result = model.generate_content(f'{food_name.capitalize()} recipe in max 100 words')
    recipe = result.text
    return recipe

st.title("Food Name and Recipe Finder")

col1, col2 = st.columns(2)

uploaded_image = col1.file_uploader("Upload an image of the food", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1.image(image, caption='Uploaded Image', use_column_width=True)
    food_name = predict_food_name(uploaded_image)
    if col1.button("Get Food Name & Recipe"):
        with st.spinner('Processing...'):
            food_name = predict_food_name(uploaded_image)
#            st.success(f"Food Name: {food_name.capitalize()}")
            st.session_state.food_name = food_name

if "food_name" in st.session_state:
    with st.spinner("Generating Recipe..."):
        food_name = st.session_state.food_name
        recipe = generate_recipe(food_name)
        col2.subheader(f"Description : {food_name.capitalize()}")
        col2.write(recipe)
