import time
from utils import constants
import pickle

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def get_data():
    dialogues = pickle.load(open(constants.DIALOGUE_PATH, 'rb'))
    labels = pickle.load(open(constants.ARTICLE_LABEL_PATH, 'rb'))
    article_sentences = pickle.load(open(constants.ARTICLE_PATH, 'rb'))
    article_to_id = pickle.load(open(constants.ID_PATH, 'rb'))
    return dialogues, labels, article_sentences, article_to_id