from collections import defaultdict
import json
from re import L
from tkinter import E
import nltk
import os
import random
import torch

import pandas as pd

from absl import app
from absl import flags
from copy import deepcopy
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from typing import List
from transformers import MarianMTModel, MarianTokenizer
from transformers import MarianTokenizer, MarianMTModel

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from tqdm import tqdm

import random
random_temp_name = os.path.join('temp', str(
    random.randint(0, 9999999999)) + '.csv')

nltk.download('stopwords')

stop_words = stopwords.words('english')

FLAGS = flags.FLAGS

flags.DEFINE_string("augmentation_dir", "...", "")
flags.DEFINE_string("dataset_path", "...", "")

flags.DEFINE_integer("batch_size", 64, "")

flags.DEFINE_boolean("backtranslation", False, "")
flags.DEFINE_integer(
    "chain_length", 1, "Chain length for backtranslation: <chain_length> - 1 is the number of intermediate languages.")

flags.DEFINE_boolean("synonym_replacement", False,
                     "Turn on for synonym replacement.")
flags.DEFINE_integer("num_synonym_replacement", 1,
                     "Number of synonyms to replace; Only works if synonym_replacement is enabled.")

flags.DEFINE_boolean("random_deletion", False, "Turn on random deletion.")
flags.DEFINE_float("prob_random_deletion", 0.1,
                   "Probability of deletion for each word. Only works if random_deletion is enabled.")

flags.DEFINE_string("augmentation_graph", "augmentation_graph.json",
                    "Augmentation graph that specifies what langauges can be used for backtranslation. If this graph is not specified, then all languages will be used.")

flags.DEFINE_integer("num_replicated_augmentations", 50,
                     "Number of augmentations to generate. Keep to at least 50 for paper results. The higher the better if computational resources permit.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################
# EDA Functions (same as EDA)
########################################################################

def synonym_replacement(words, n, emotion_words=None):
    new_words = words.copy()
    random_word_list = list(
        set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        if emotion_words and random_word in emotion_words:
            synonyms = get_synonyms(random_word)
        elif emotion_words and random_word not in emotion_words:
            continue
        else:
            synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word ==
                         random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join(
                [char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def random_deletion(words, p, emotion_words=None):

    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        if emotion_words:
            if word in emotion_words:
                new_words.append(word)
                continue
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


possible_languages = ['en', 'fr', 'de', 'it', 'ro', 'uk', 'es',
                      'hu', 'fi', 'toi', 'tll', 'swc', 'sv', 'wls', 'tw', 'tum']


def build_backtranslation_graph():

    augmentation_graph = defaultdict(list)
    for i in range(len(possible_languages)):
        for j in range(i + 1, len(possible_languages)):
            if i != j:
                try:
                    model_name = f"Helsinki-NLP/opus-mt-{possible_languages[i]}-{possible_languages[j]}"
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    augmentation_graph[possible_languages[i]].append(
                        possible_languages[j])
                except Exception as e:
                    print(e)
                    print(possible_languages[i], possible_languages[j])
                try:
                    model_name = f"Helsinki-NLP/opus-mt-{possible_languages[j]}-{possible_languages[i]}"
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    augmentation_graph[possible_languages[j]].append(
                        possible_languages[i])
                except Exception as e:
                    print(e)
                    print(possible_languages[j], possible_languages[i])

    return augmentation_graph


def translate_single_language(src, trg, dataframe, batch_size):

    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ids = dataframe['Id'].tolist()
    texts = dataframe['Text'].tolist()
    labels = dataframe['Label'].tolist()
    strengths = dataframe['Strength'].tolist()

    translated_dataframe = []

    num_batches = len(texts) // batch_size

    model.to(device)

    for i in tqdm(range(num_batches + 1)):
        if i * batch_size >= len(texts):
            sample_texts = texts[(i * batch_size):]
            sample_ids = ids[(i * batch_size):]
            sample_labels = labels[(i * batch_size):]
            sample_strengths = strengths[(i * batch_size):]
        else:
            sample_texts = texts[(i * batch_size):((i + 1) * batch_size)]
            sample_ids = ids[(i * batch_size):((i + 1) * batch_size)]
            sample_labels = labels[(i * batch_size):((i + 1) * batch_size)]
            sample_strengths = strengths[(
                i * batch_size):((i + 1) * batch_size)]

        batch = tokenizer(sample_texts, return_tensors="pt",
                          padding=True, truncation=True).to(device)
        gen = model.generate(**batch, max_length=128)
        new_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        zipped = list(zip(sample_ids, new_texts,
                      sample_labels, sample_strengths))
        translated_dataframe += zipped

    translated_dataframe = pd.DataFrame(translated_dataframe, columns=[
                                        'Id', 'Text', 'Label', 'Strength'])
    translated_dataframe.to_csv(random_temp_name, index=False)
    return pd.read_csv(random_temp_name)


def backtranslate(dataframe, batch_size, augmentation_graph, chain_length, idx):

    current_language = "en"
    current_dataframe = dataframe

    if not os.path.isdir('temp'):
        os.mkdir('temp')

    for _ in range(chain_length - 1):
        possible_languages = augmentation_graph[current_language]
        chosen_langauge = random.choice(possible_languages)
        print('Translating from', current_language, 'to', chosen_langauge)
        current_dataframe = translate_single_language(
            current_language, chosen_langauge, current_dataframe, batch_size)
        current_language = chosen_langauge

    while True:
        possible_languages = augmentation_graph[current_language]
        if 'en' not in possible_languages:
            chosen_langauge = random.choice(possible_languages)
            print('Translating from', current_language, 'to', chosen_langauge)
            current_dataframe = translate_single_language(
                current_language, chosen_langauge, current_dataframe, batch_size)
            current_language = chosen_langauge
        else:
            print('Translating from', current_language, 'to', 'en')
            current_dataframe = translate_single_language(
                current_language, 'en', current_dataframe, batch_size)
            break

    return pd.read_csv(random_temp_name)


def eda_augment_with_pool(dataframe, pool, emotion_set):

    random.shuffle(pool)
    new_dataframe = []
    for i, row in dataframe.iterrows():
        text = row['Text']
        for crt_augmentation in pool:
            text = ' '.join(crt_augmentation[0](
                text.split(), crt_augmentation[-1]))
        new_dataframe.append([row['Id'], text, row['Label'], row['Strength']])

    new_dataframe = pd.DataFrame(new_dataframe, columns=dataframe.columns)
    return new_dataframe


def main(argv):

    if not os.path.exists(FLAGS.augmentation_graph):
        aug_graph = build_backtranslation_graph()
        with open(FLAGS.augmentation_graph, 'w') as f:
            json.dump(aug_graph, f)
    else:
        with open(FLAGS.augmentation_graph) as f:
            aug_graph = json.load(f)

    train_dataframe = pd.read_csv(FLAGS.dataset_path)

    augmentation_mapper = {
        "synonym_replacement": (synonym_replacement, FLAGS.synonym_replacement, FLAGS.num_synonym_replacement),
        "random_deletion": (random_deletion, FLAGS.random_deletion, FLAGS.prob_random_deletion),
    }

    if not os.path.isdir(FLAGS.augmentation_dir):
        os.mkdir(FLAGS.augmentation_dir)
    if FLAGS.chain_length < 2:
        FLAGS.chain_length = 1
    augmentation_strength = FLAGS.num_synonym_replacement + \
        int(FLAGS.prob_random_deletion * 20) + 3 * (FLAGS.chain_length - 1)
    augmentation_pool = []
    for k in augmentation_mapper:
        if augmentation_mapper[k][1] == True:
            augmentation_pool.append(augmentation_mapper[k])
    train_dataframe['Strength'] = [
        augmentation_strength] * len(train_dataframe)
    for i in range(1, FLAGS.num_replicated_augmentations + 1):
        working_dataframe = deepcopy(train_dataframe)
        if len(augmentation_pool) != 0:
            working_dataframe = eda_augment_with_pool(
                working_dataframe, augmentation_pool, None)
        if FLAGS.chain_length >= 2:
            working_dataframe = backtranslate(
                working_dataframe, FLAGS.batch_size, aug_graph, FLAGS.chain_length, i)

        working_dataframe.to_csv(os.path.join(FLAGS.augmentation_dir, 'augmentation' + '_' + str(FLAGS.num_synonym_replacement) +
                                 '_' + str(FLAGS.prob_random_deletion) + '_' + str(FLAGS.chain_length) + '_' + str(i)), index=False)


if __name__ == "__main__":
    app.run(main)
