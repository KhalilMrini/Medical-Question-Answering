WIKI_PATH = '/vault/datasets/wikipedia/docs.db'
PATH = '/vault/datasets/khalil/ParlAI/data/wizard_of_wikipedia/'

# Dataset paths
TRAIN_PATH = 'train.json'
VALID_SEEN_PATH = 'valid_random_split.json'
VALID_UNSEEN_PATH = 'valid_topic_split.json'
TEST_SEEN_PATH = 'test_random_split.json'
TEST_UNSEEN_PATH = 'test_topic_split.json'
DATASET_PATHS = [TRAIN_PATH, VALID_SEEN_PATH, VALID_UNSEEN_PATH, TEST_SEEN_PATH, TEST_UNSEEN_PATH]

# Output file paths
DIALOGUE_PATH = PATH + 'dialogues.pkl'
ENCODED_DIALOGUE_PATH = PATH + 'dialogues_encoded.pt'
ARTICLE_LABEL_PATH = PATH + 'article_labels.pt'
ARTICLE_PATH = PATH + 'article_sentences.pkl'
ARTICLE_TITLE_PATH = PATH + 'article_titles.pkl'
ENCODED_ARTICLE_PATH = PATH + 'article_sentences_encoded_'
ENCODED_ARTICLE_TYPE = '.json'
ENCODED_ARTICLE_PT_PATH = PATH + 'article_sentences_encoded.pt'
ENCODED_ARTICLE_RED_PATH = PATH + 'article_sentences_reduced.pt'
ID_PATH = PATH + 'article_to_id.pkl'
SELECTOR_DATA_PATH = PATH + 'selector_data.json'