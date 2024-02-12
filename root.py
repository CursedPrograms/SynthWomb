import pickle
from scripts.play_audio import play_audio
from scripts.gpt.system.generate_text import generate_text
from scripts.gpt.system.clean_text import clean_text

# Cache functions
functions_to_cache = {
    'play_audio': play_audio,
    'generate_text': generate_text,
    'clean_text': clean_text,
}

with open('cached_functions.pkl', 'wb') as file:
    pickle.dump(functions_to_cache, file)