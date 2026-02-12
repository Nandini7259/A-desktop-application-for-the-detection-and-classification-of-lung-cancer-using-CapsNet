import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pymysql
import re

# ------------------------
# 1. Database Connection
# ------------------------
def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",  # Replace with your WAMP username
        password="",  # Replace with your WAMP password
        database="lung_cancer_app",
        cursorclass=pymysql.cursors.DictCursor,
    )

# ------------------------
# 2. Database Functions
# ------------------------
def create_user(username, password):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
            cursor.execute(sql, (username, password))
            conn.commit()
        return True
    except pymysql.err.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM users WHERE username = %s AND password = %s"
            cursor.execute(sql, (username, password))
            return cursor.fetchone() is not None
    finally:
        conn.close()

# ------------------------
# 3. Password Validation
# ------------------------
def is_valid_password(password):
    # Check if password contains at least one letter, one number, and one special character
    return bool(re.match(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>])[A-Za-z\d!@#$%^&*(),.?":{}|<>]{8,}$', password))

# ------------------------
# 4. Streamlit Authentication Pages with Validation
# ------------------------
def signup():
    st.title("Sign Up Page")
    new_username = st.text_input("Create a Username")
    new_password = st.text_input("Create a Password", type="password")
    signup_button = st.button("Sign Up")

    # Validate Sign Up Form
    if signup_button:
        if not new_username or not new_password:
            st.error("Both username and password are required.")
        elif len(new_password) < 8:
            st.error("Password must be at least 8 characters long.")
        elif not is_valid_password(new_password):
            st.error("Password must contain at least one letter, one number, and one special character.")
        elif create_user(new_username, new_password):
            st.success("Account created successfully!")
            st.session_state["logged_in"] = True
            st.session_state["page"] = "main"
        else:
            st.error("Username already exists. Please choose another.")

def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    signup_redirect = st.button("Sign Up Instead")

    # Validate Login Form
    if login_button:
        if not username or not password:
            st.error("Both username and password are required.")
        elif not is_valid_password(password):
            st.error("Password must be alphanumeric and contain at least one special character.")
        elif authenticate_user(username, password):
            st.success("Login successful!")
            st.session_state["logged_in"] = True
            st.session_state["page"] = "main"
        else:
            st.error("Invalid username or password.")
    elif signup_redirect:
        st.session_state["page"] = "signup"

# ------------------------
# 5. Main Application Logic
# ------------------------
def main_app():
    # Load the trained model
    MODEL_PATH = 'capsnet_lung_cancer_model.h5'
    model = tf.keras.models.load_model(MODEL_PATH)

    CLASS_NAMES = ['Stage1', 'Stage2', 'Stage3', 'Normal']

    st.title("Lung Cancer Stage Detection")
    st.write("Upload a chest CT image, and the model will predict the lung cancer stage.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CT image...", type=["jpg", "png"])

    # Prediction logic
    def predict_image(image_path):
        img_size = (128, 128)  # Ensure this matches the training image size
        img = load_img(image_path, target_size=img_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]
        
        return CLASS_NAMES[predicted_class], confidence

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image("temp_image.png", caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        st.write("Result")
        predicted_label, confidence = predict_image("temp_image.png")
        
        # Display result
        st.success(f"Prediction: **{predicted_label}** (Confidence: {confidence:.2f})")
    else:
        st.info("Please upload a CT image to get started.")

    st.write("Model trained using a Capsule Network to classify lung cancer stages.")

# ------------------------
# 6. App State Management
# ------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "page" not in st.session_state:
    st.session_state["page"] = "login"

# Handle navigation based on session state
if st.session_state["page"] == "login":
    login()
elif st.session_state["page"] == "signup":
    signup()
elif st.session_state["page"] == "main" and st.session_state["logged_in"]:
    main_app()
else:
    st.error("Something went wrong. Please refresh the page.")
