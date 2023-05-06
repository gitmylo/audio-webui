import bark.generation
from patches.bark_generation import *


def patch():
    bark.generation.generate_text_semantic = generate_text_semantic
