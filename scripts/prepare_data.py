import os
import csv
import itertools
import functools
import tensorflow as tf
import numpy as np
import array

tf.flags.DEFINE_integer(
    "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum sentence length")

tf.flags.DEFINE_string(
    "input_dir", os.path.abspath("../../data"),
    "Input directory containing original CSV data files (default = '../../data')")

tf.flags.DEFINE_string(
    "output_dir", os.path.abspath("../../data"),
    "Output directory for TFrEcord files (default = '../../data')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.csv")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")

def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)

def create_csv_iter(filename):
    """
    Returns an iterator over a csv file. Skips the header.
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header
        next(reader)
        for row in reader:
            yield row

def create_vocab(input_iter, min_frequency):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary for the input iterator
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_dir)
    return vocab_processor

def transform_sentence(sequence, vocab_processor):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array
    """
    return next(vocab_processor.transform([sequence])).tolist()

def create_text_sequence_feature(fl, sentence, sentence_len, vocab):

