<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/python%20-%231F5B94.svg?&style=for-the-badge&logo=python&logoColor=white"/>
    <img alt="HTML5" src="https://img.shields.io/badge/html5%20-%231F5B94.svg?&style=for-the-badge&logo=html5&logoColor=white"/>
  <img alt="CSS3" src="https://img.shields.io/badge/css3%20-%231F5B94.svg?&style=for-the-badge&logo=css3&logoColor=white"/> 
</div>

<div align="center">
    <img alt="TensorFlow" src="https://img.shields.io/badge/tensorflow%20-%231F5B94.svg?&style=for-the-badge&logo=tensorflow&logoColor=white"/>
    <img alt="Flask" src="https://img.shields.io/badge/flask%20-%231F5B94.svg?&style=for-the-badge&logo=flask&logoColor=white"/>
  <img alt="OpenCV" src="https://img.shields.io/badge/opencv-%231F5B94.svg?&style=for-the-badge&logo=opencv&logoColor=white"/>
    <img alt="CSS3" src="https://img.shields.io/badge/css3%20-%231F5B94.svg?&style=for-the-badge&logo=css3&logoColor=white"/> 
</div>
<br>
<div align="center">
<a href="https://github.com/SynthWomb" target="_blank" align="center">
    <img src="https://github.com/SynthWomb/synth.womb/blob/main/logos/synthwomb07.png"
        alt="SynthWomb">
</a>
</div>
<br>

The SynthWomb, or Synthia, seamlessly integrates multiple technologies in a unified project. Powered by the Hugging Face Transformers library, it engages with a pre-trained GPT-2 model. TensorFlow ensures efficient model management, while AutoTokenizer streamlines tokenization for optimal performance. Users interactively input prompts, prompting the GPT-2 model to generate text responses. To refine the user experience, extraneous warning messages tied to the transformers library are intelligently silenced.

Expanding its capabilities, the project integrates the DeepFace library with OpenCV to extract gender information from input data, presenting gender predictions visually. The SDXL-Turbo model is harnessed for diverse image processing tasks. The project's command-line interface facilitates both Text-to-image and Image-to-image operations, empowering users to seamlessly select operations, input prompts, and witness dynamic script execution.

This versatile Python codebase introduces a Generative Adversarial Network (GAN) tailored for training and generating AI art. This addition brings an artistic dimension to the project's repertoire, enhancing its overall capabilities.

<br>
<div align="center">
<a href="https://github.com/SynthWomb" target="_blank" align="center">
    <img src="https://github.com/SynthWomb/synth.womb/blob/main/logos/synthwomb-promo01.png"
        alt="SynthWomb">
</a>
</div>
<br>

## Scripts:

- **main.py:** The selection menu for SynthWomb
- **app.py:** Run SynthiaGPT as a web-server

### /scripts

- **video_encoder.py:** Encode a video using SynthWombAI
- **install_dependencies.py:** Install dependencies
- **image_generator.py:** Generate AI art
- **image_generator.py:** Display webcam

### /scripts/ai_trainer

- **trainer.py:** Runs SynthWombAI trainer
- **modelout.py:** Output images from trained models with SynthWomb
- **image-processor.py:** Prepare images for SynthWombAI
- **preprocessor_data.py:** Dependency for SynthWombAI /trainer

### /scripts/gender_age_prediction

- **gender_predictor.py:** Predicts gender from an input
- **gender_age_predictor.py:** Predicts gender and age from an input
- **gender_age_predictor_webcam.py:** Predicts gender and age from a webcam feed

### /scripts/gpt

- **transformer.py:** Run SynthiaGPT
- **transformer_t2s.py:** Run SynthiaGPT with text-to-speech functionality
- **transformer_s2t2s.py:** Run SynthiaGPT with speech-to-text-to-speech functionality
- **transformer_webcam.py:** Run SynthiaGPT with webcam functionality

### /scripts/gpt/system

- **generate_text.py:** The GPT text generator
- **clean_text.py:** Clean text from recorded voice audio
- **stt.py:** Speech-to-text
- **tts.py:** Text-to-speech
- **capture_photo.py:** Capture photo with webcam
- **play_audio.py:** Play generated audio

<br>
<div align="center">
<a href="" target="_blank">
    <img src="https://github.com/SynthWomb/Synthia/raw/main/demo_images/synthia00-cover.png"
        alt="SynthiaGPT">
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
pygame==2.5.2
```
<br>

## How to Run:

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

<br>
<div align="center">
<a href="" target="_blank">
    <img src="https://github.com/SynthWomb/SynthWomb/blob/main/logos/screenshot.png"
        alt="SynthiaGPT">
</a>
</div>
<br>

- [GloriosaAI](https://github.com/CursedPrograms/GloriosaAI)
- [Gender-Age-ID](https://github.com/CursedPrograms/Gender-Age-ID)
- [Detect-Face](https://github.com/CursedPrograms/Detect-Face)
- [Cursed-GPT](https://github.com/CursedPrograms/Cursed-GPT)
- [Image-Generator](https://github.com/CursedPrograms/Image-Generator)

<br>
<div align="center">
<a href="https://github.com/SynthWomb" target="_blank" align="center">
    <img src="https://github.com/SynthWomb/synth.womb/blob/main/logos/synthwomb07.png"
        alt="SynthWomb" style="width:200px;"/>
</a>
    <br>
</div>
<div align="center">
<a href="https://github.com/CursedPrograms" target="_blank">
    <img src="https://github.com/CursedPrograms/cursedentertainment/raw/main/images/logos/logo-wide-grey.png"
        alt="CursedEntertainment Logo" style="width:250px;">
</a>
</div>
