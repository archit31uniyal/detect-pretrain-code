import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
        self.parser.add_argument('--ref_model', type=str, default="huggyllama/llama-7b")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--data', type=str, default="swj0419/WikiMIA", help="the dataset to evaluate")
        self.parser.add_argument('--non_member_data', type=str, default=None, help="Non-member dataset path")
        self.parser.add_argument('--length', type=int, default=64, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, parapgrase")
        self.parser.add_argument('--overwrite_pickle', action="store_true", help="Flag to overwrite pickle files")
        self.parser.add_argument('--generate', action="store_true", help="Flag to generate sequences")
        self.parser.add_argument('--simcse', action="store_true", help="Flag to calculate SimCSE score")
        self.parser.add_argument('--prompt', action="store_true", help="Flag to enable prompting model for generations")
        self.parser.add_argument('--task', type=str, default="gen", help="Tasks: gen, trivia")
        self.parser.add_argument('--qna', action="store_true", help="Flag to enable QnA mode")
        self.parser.add_argument('--simcse_task', type=str, default=None, help="Tasks: gen, ref_gen, prompt_gen, prompt_long_gen, qna_trivia, qna_ref_trivia")
        self.parser.add_argument('--score_type', type=str, default='unlearn', help="Type for calculating threshold for unlearned model, pretrained model or ratio of both. Options: unlearn, ref, ratio")
        




