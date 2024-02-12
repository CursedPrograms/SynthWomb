<div align="center">
<a href="https://github.com/SynthWomb" target="_blank" align="center">
    <img src="https://github.com/SynthWomb/synth.womb/blob/main/logos/synthwomb07.png"
        alt="SynthWomb">
</a>
</div>
<br>
The SynthWomb or Synthia project leverages the Hugging Face Transformers library to interact with a pre-trained GPT-2 model. TensorFlow is employed for effective model management, and AutoTokenizer ensures efficient tokenization. The script allows users to interactively input prompts, generating text responses from the GPT-2 model. To enhance the user experience, unnecessary warning messages related to the transformers library are intelligently suppressed.

Additionally, the project incorporates the DeepFace library in conjunction with OpenCV to extract gender information from input data, with the results of gender prediction being displayed. Furthermore, the SDXL-Turbo model is utilized for diverse image processing tasks. The project offers a straightforward command-line interface, enabling both Text-to-image and Image-to-image operations. Users can easily select an operation, input prompts, and witness the script dynamically generating and executing code snippets.

This versatile Python codebase includes a Generative Adversarial Network (GAN) designed for the training and generation of AI art, adding an artistic dimension to the project's capabilities.
<br>
<div align="center">
<a href="https://github.com/SynthWomb" target="_blank" align="center">
    <img src="https://github.com/SynthWomb/synth.womb/blob/main/logos/synthwomb-promo.png"
        alt="SynthWomb">
</a>
</div>
<br>

## Requirements:

<br>

```bash
transformers==4.37.1
torch==1.10.0
torchvision==0.16.1
torchaudio==2.1.1
SpeechRecognition==3.10.1
tensorflow==2.14.0
numpy== 1.26.2
matplotlib==3.8.2
Pillow==10.1.0
opencv-python==4.8.1.78
deepface==0.0.81
flask==3.0.0
gtts==2.5.0
pyaudio==0.2.14 
pydub==0.25.1
beautifulsoup4==4.10.0
SpeechRecognition==3.10.1
```
<br>

# How to Run

```bash
pip install -r requirements.txt
```
```bash
pip install torch torchvision torchaudio
```
```bash
python main.py
```
```bash
python app.py
```
<br>

To make the setup script executable, run the following command in your terminal:

```bash
chmod +x setup.sh
```
