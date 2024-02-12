from flask import Flask, render_template, request
import os
import shutil
import transformers
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
import logging
from scripts.gpt.system.generate_text import generate_text
import webbrowser

transformers.logging.set_verbosity_error()
tf.get_logger().setLevel(logging.ERROR)

app = Flask(__name__, static_url_path='/static')

model_name = "gpt2"
model = TFAutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token_id=50256)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/image-generator')
def image_generator():
    return render_template('image-generator.html')

@app.route('/synthia-gpt')
def synthia_gpt():
    return render_template('synthia-gpt.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_text = generate_text(prompt, model, tokenizer)
    return render_template('synthia-gpt.html', prompt=prompt, generated_text=generated_text)

def copy_newest_images(output_directory, target_directory, num_images=10):
    # Get the list of generated images sorted by modification time
    generated_images = sorted(os.listdir(output_directory), key=lambda f: os.path.getmtime(os.path.join(output_directory, f)))

    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Copy the 10 newest images to the target directory
    for image in generated_images[-num_images:]:
        source_path = os.path.join(output_directory, image)
        target_path = os.path.join(target_directory, image)
        shutil.copyfile(source_path, target_path)

@app.route('/text_to_image', methods=['POST'])
def text_to_image():
    prompt = request.form.get('prompt')
    output_directory = 'output/generated_images'
    os.makedirs(output_directory, exist_ok=True)

    counter = 1
    output_filename = f'output_{counter}.jpg'
    while os.path.exists(os.path.join(output_directory, output_filename)):
        counter += 1
        output_filename = f'output_{counter}.jpg'

    text_to_image_code = f"""
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16")
pipe.to("cpu")
prompt = "{prompt}"
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
output_directory = 'output/generated_images'
os.makedirs(output_directory, exist_ok=True)
image.save(os.path.join(output_directory, '{output_filename}'))
    """

    print("Generated code snippet:")
    print(text_to_image_code)

    exec(text_to_image_code)

    # Copy the 10 newest images to the static/generated_images folder
    copy_newest_images(output_directory, 'static/generated_images', num_images=10)

    return render_template('image-generator.html', result_image=output_filename)

@app.route('/image_to_image', methods=['POST'])
def image_to_image():
    prompt = request.form.get('prompt')
    output_directory = 'output/generated_images'
    os.makedirs(output_directory, exist_ok=True)

    counter = 1
    output_filename = f'output_{counter}.jpg'
    while os.path.exists(os.path.join(output_directory, output_filename)):
        counter += 1
        output_filename = f'output_{counter}.jpg'

    image_to_image_code = f"""
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import Image
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16")
pipe.to("cpu")
image_path = "input/input.jpg"  
init_image = load_image(image_path).resize((512, 512))
prompt = "{prompt}"
image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
output_directory = 'output/generated_images'
os.makedirs(output_directory, exist_ok=True)
image.save(os.path.join(output_directory, '{output_filename}'))
    """

    print("Generated code snippet:")
    print(image_to_image_code)

    exec(image_to_image_code)

    copy_newest_images(output_directory, 'static/generated_images', num_images=10)
    return render_template('image-generator.html', result_image=output_filename)

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True, use_reloader=False)
