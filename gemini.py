import os
import time
import pyautogui
import pytesseract
from PIL import Image
import google.generativeai as genai
import spacy
import tkinter as tk
import re
import cv2
import numpy as np
from threading import Thread

# Configure the Gemini API with your API key
genai.configure(api_key="AIzaSyAtefeL7lHpYkivkAnSlF34JsDhC6_fs_c")

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the image using OpenCV
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.medianBlur(image, 3)  # Apply median blur with a smaller kernel
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
    return image

# Function to deskew the image
def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Function to crop the image to focus on the question and options
def crop_image(image):
    height, width = image.shape
    top = int(height * 0.1)  # Adjusted cropping parameters
    bottom = int(height * 0.9)
    left = int(width * 0.1)
    right = int(width * 0.9)
    cropped_image = image[top:bottom, left:right]
    return cropped_image

# Function to take a screenshot and process it
def process_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    image = preprocess_image("screenshot.png")
    cv2.imwrite("preprocessed_image.png", image)  # Save preprocessed image for debugging
    cropped_image = crop_image(image)
    cv2.imwrite("cropped_image.png", cropped_image)  # Save cropped image for debugging
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(cropped_image, config=custom_config)
    print("Extracted Text:", text)  # Debugging print statement
    return text

# Function to clean and extract question and options
def extract_question_and_options(text):
    question_match = re.search(r'(.+?\?)', text)
    options_match = re.findall(r'(\d+\.\s?.+|[a-dA-D]\.\s?.+|[•*]\s?.+|[Oo©]\s?.+)', text)
    if question_match and options_match:
        question = question_match.group(1)
        options = ' '.join(options_match[:4])  # Ensure only the first four options are considered
        return f"{question} {options}"
    return None

# Function to extract question and fetch answer
def get_answer(text):
    cleaned_text = extract_question_and_options(text)
    if cleaned_text:
        prompt = f"{cleaned_text} Choose the correct option without any explanation. Just provide the option."
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        print("Generated Answer:", response.text.strip())  # Debugging print statement
        return response.text.strip()
    return "Unable to extract question and options."

# Function to display the answer on the screen
def display_answer(answer):
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry("+100+100")
    root.attributes("-topmost", True)  # Ensure the window stays on top
    label = tk.Label(root, text=answer, font=("Helvetica", 16), bg="yellow")
    label.pack()
    root.after(5000, root.destroy)  # Display for 5 seconds
    root.mainloop()

# Function to start the main loop in a separate thread
def start_processing():
    global running
    running = True
    while running:
        text = process_screenshot()
        answer = get_answer(text)
        display_answer(answer)
        time.sleep(5)  # Check every 5 seconds

# Function to stop the processing loop
def stop_processing():
    global running
    running = False

# Function to start the processing loop
def start_button_click():
    processing_thread = Thread(target=start_processing)
    processing_thread.start()

# Create the main application window
app = tk.Tk()
app.title("MCQ Answer Finder")
app.geometry("400x300")

start_button = tk.Button(app, text="Start", command=start_button_click)
start_button.pack()

stop_button = tk.Button(app, text="Stop", command=stop_processing)
stop_button.pack()

app.mainloop()
