import os
import subprocess
import speech_recognition as sr
from gtts import gTTS
import json

def read_config():
    with open('settings.json', 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

def greet_user():
    config = read_config()
    robot_name = config.get("robot_name", "")
    print(f"Welcome to {robot_name}.")
    greeting_text = f"Greetings, human. I am {robot_name}. How may I assist you today?"
    os.makedirs('output', exist_ok=True)
    tts = gTTS(text=greeting_text, lang='en')
    tts.save('output/greeting.mp3')
    os.system("start output/greeting.mp3")

def main():
    greet_user()

    scripts = {
        "1": {
            "name": "Run ai_trainer/'main.py'",
            "description": "Train an image GAN, encode videos and output .h5 models",
            "file_name": "scripts/ai_trainer/main.py",
            "phrases": ["one", "image GAN trainer"]
        },
        "2": {
            "name": "Run gender_age_prediction/'main.py'",
            "description": "Predict age and gender of a given image",
            "file_name": "scripts/gender_age_prediction/main.py",
            "phrases": ["two", "gender age prediction"]
        },
        "3": {
            "name": "Run gpt/'main.py'",
            "description": "Run a GPT, that includes a Text-to-Speech module to make it talk",
            "file_name": "scripts/gpt/main.py",
            "phrases": ["three", "GPT"]
        },
        "4": {
            "name": "Run 'detect_face.py'",
            "description": "Detect a face in an image",
            "file_name": "scripts/face_detection/main.py",
            "phrases": ["four", "detect face"]
        },
        "5": {
            "name": "Run 'image_generator.py'",
            "description": "Generate an Image with SDXL",
            "file_name": "scripts/image_generator.py",
            "phrases": ["five", "image generator"]
        },
        "6": {
            "name": "Run 'video_encoder.py'",
            "description": "Encode a video using images frames",
            "file_name": "scripts/video_encoder.py",
            "phrases": ["six", "video encoder"]
        },
        "00": {
            "name": "Run 'install_dependencies.py'",
            "description": "Install dependencies",
            "file_name": "scripts/install_dependencies.py",
            "phrases": ["zero zero", "null null", "install dependencies"]
        },
    }

    while True:
        print("\nAvailable Scripts:")
        for key, script_info in scripts.items():
            print(f"{key}: {script_info['name']} - {script_info['description']}")

        print("You can also say 'quit' to exit.")

        input_method = input("Choose input method (1 for keyboard, 2 for voice, 'q' to quit): ")

        if input_method == '1':
            user_input = input("Enter the script number or a phrase: ").strip().lower()
            matched_script = next((script for script, v in scripts.items() if user_input in (script, *v.get("phrases", []))), None)
            user_choice = matched_script if matched_script else user_input
        elif input_method == '2':
            user_choice = speech_to_text().lower()
            matched_script = next((script for script, v in scripts.items() if user_choice in (script, *v.get("phrases", []))), None)
            user_choice = matched_script if matched_script else user_choice
        elif input_method.lower() == 'q':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter '1', '2', or 'q'.")
            continue

        if user_choice.lower() == 'quit':
            break

        if user_choice in scripts:
            selected_script = scripts[user_choice]
            script_file_name = selected_script["file_name"]
            script_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_file_name)

            if os.path.exists(script_file_path):
                try:
                    subprocess.run(["python", script_file_path])
                except Exception as e:
                    print(f"An error occurred while running the script: {e}")
            else:
                print(f"Script file '{script_file_name}' does not exist.")
        else:
            print("Invalid choice. Please select a valid script number or phrase.")

def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        choice = recognizer.recognize_google(audio)
        print(f"You said: {choice}")
        return choice
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Error with the speech recognition service; {e}")
        return ""

if __name__ == "__main__":
    main()
