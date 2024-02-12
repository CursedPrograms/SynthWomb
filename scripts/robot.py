import cv2
from deepface import DeepFace
import numpy as np
import transformers
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import subprocess
import tensorflow as tf
import logging
import os
from bs4 import BeautifulSoup
import time
import numpy as np
import pyaudio
import speech_recognition as sr
import wave
from play_audio import play_audio
from ..scripts.gpt.system.generate_text import generate_text
from ..scripts.gpt.system.clean_text import clean_text
import sys

transformers.logging.set_verbosity_error()
tf.get_logger().setLevel(logging.ERROR)

import urllib.request

sys.stdout = open(os.devnull, 'w')

def download_file(url, save_path):
    urllib.request.urlretrieve(url, save_path)

def download_caffe_model_files():
    # Create a directory to save the model files
    os.makedirs("models", exist_ok=True)

    # Download deploy.prototxt
    deploy_prototxt_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
    deploy_prototxt_path = os.path.join("models", "deploy.prototxt")
    download_file(deploy_prototxt_url, deploy_prototxt_path)

    # Download mobilenet_iter_73000.caffemodel
    caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    caffemodel_path = os.path.join("models", "mobilenet_iter_73000.caffemodel")
    download_file(caffemodel_url, caffemodel_path)

def extract_gender_and_age(frame, face_cascade):
    try:
        # Convert the frame to RGB (required by DeepFace)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use Haar Cascade to detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]

            # Use DeepFace to analyze gender and age
            result = DeepFace.analyze(face_roi, actions=['gender', 'age'])

            gender_stats = result[0]['gender']
            age = int(result[0]['age'])
            print("Predicted Gender:", gender_stats)
            print("Predicted Age:", age)

            # Select the dominant gender label
            gender = max(gender_stats, key=gender_stats.get)

            # Display gender and age information on the frame
            cv2.putText(frame, f"Gender: {gender} - {gender_stats[gender]:.2f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Age: {age} years", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

    except Exception as e:
        print(f"Error: {e}")
        return frame

def capture_audio(duration=30, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    print("Recording audio...")

    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Save audio data to a temporary file
    temp_audio_file = "output/audio/temp_audio.wav"
    os.makedirs(os.path.dirname(temp_audio_file), exist_ok=True)
    wf = wave.open(temp_audio_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return temp_audio_file

def text_to_speech(text):
    cleaned_text = clean_text(text)
    
    # Generate a unique filename with a timestamp
    timestamp = int(time.time())
    audio_path = f"output/audio/output_{timestamp}.mp3"

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    tts = gTTS(text=cleaned_text, lang='en', slow=False)
    tts.save(audio_path)
    play_audio(audio_path)

def capture_photo(output_dir="output/shot"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Capture a single frame
    ret, frame = cap.read()

    # Release the webcam
    cap.release()

    # Save the captured frame to the specified directory
    image_path = os.path.join(output_dir, "captured_photo.jpg")
    cv2.imwrite(image_path, frame)

    return image_path

def image_description(image_path):
    # Load the pre-trained MobileNet SSD model and its class labels
    net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/mobilenet_iter_73000.caffemodel")
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]

    # Load and preprocess the input image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections and get the description of the most confident prediction
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            description = f"A photo containing a {classes[class_id]} with confidence {confidence:.2f}"
            return description

    return "Unable to identify the content of the photo."

def text_to_image(prompt):
    print("Using SDXL-Turbo for Text-to-image:")
    print("Make sure to install the required packages using:")
    print("pip install diffusers transformers accelerate --upgrade")
    print()

    output_directory = 'output/generated_images'
    os.makedirs(output_directory, exist_ok=True)

    # Use a counter to create a unique filename
    counter = 1
    output_filename = f'output_image_{counter}.jpg'
    while os.path.exists(os.path.join(output_directory, output_filename)):
        counter += 1
        output_filename = f'output_image_{counter}.jpg'

    text_to_image_code = f"""
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16")
pipe.to("cpu")
prompt = "{prompt}"
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save(os.path.join(output_directory, '{output_filename}'))
    """

    print("Generated code snippet:")
    print(text_to_image_code)

    exec(text_to_image_code)

def main():
    # Load GPT-2 model for text generation
    model_name = "gpt2"
    gpt_model = TFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token_id=50256)

    # Download Caffe model files
    download_caffe_model_files()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Speech recognizer
    recognizer = sr.Recognizer()

    while True:
        # Capture video from webcam
        ret, frame = cap.read()

        # Check if frame is valid
        if not ret:
            print("Error: Couldn't capture frame from the webcam.")
            break

        # Apply face detection and gender/age analysis to the frame
        frame = extract_gender_and_age(frame, face_cascade)

        # Display the processed frame
        cv2.imshow('Webcam Feed with Gender and Age Analysis', frame)

        # Save the captured frame to the specified directory
        captured_image_path = "output/shot/captured_photo.jpg"
        cv2.imwrite(captured_image_path, frame)

        # Capture audio and convert to text
        audio_text = ""
        with sr.AudioFile(capture_audio()) as source:
            try:
                audio = recognizer.record(source)
                audio_text = recognizer.recognize_google(audio)
                print(f"Audio Text: {audio_text}")
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service; {e}")

        # Construct prompt based on gender, age, audio, and image description
        gender_prompt = "man" if "man" in frame else "woman" if "woman" in frame else "person"
        age_prompt = "33"  # Replace with the detected age
        image_desc = image_description(captured_image_path)

        prompt = f"A {gender_prompt} {age_prompt}, {audio_text}, {image_desc}"

        # Generate text based on the prompt
        generated_text = generate_text(prompt, gpt_model, tokenizer)

        # Print and say the generated text
        if generated_text:
            print("Generated Text:")
            print(generated_text)
            text_to_speech(generated_text)

        # Draw an image based on the prompt
        if prompt:
            text_to_image(prompt)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.stdout = sys.__stdout__
    main()
