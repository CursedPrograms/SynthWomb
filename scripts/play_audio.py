import pygame
import time

def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)

    play_count = 0
    while play_count < 1:
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        play_count += 1