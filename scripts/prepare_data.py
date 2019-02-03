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
    """
    Writes a sentece to FeatureList protocol buffer
    """
    sentence_trasformed = transform_sentence(sentece, vocab)
    for word_id in sentence_trasformed:
        fl.feature.add().int64_list.value.extend([word_id])
    return fl

def create_example_train(row, vocab):
    """
    Creates a training example for the Ubuntu Dialog Corpus dataset.
    Returns a tensorflow. Example Protocol Buffer object.
    """

    context, utterance, label = row
    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(utterance, vocab)
    context_len = len(next(vocab._tokenizer([context])))
    utterance_len = len(next(vocab._tokenizer([utterance])))
    label = int(float(label))

    # New Example 
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
    example.features.feature["label"].int64_list.value.extend([label])
    return example


def create_example_test(row, vocab):
    """
    Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
    Returns a tensorflow. Example Protocol Buffer Object
    """
    context, utterance = row[:2]
    distractions = row[2:]
    context_len = len(next(vocab._tokenizer([context])))
    utterance_len = len(next(vocab._tokenizer([utterance])))
    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(utterance, vocab)

    # New Example 
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])

    # Distraction sequences 
    for i, distractor in enumerate(distractions):
        dis_key = "distractor_{}".format(i)
        dis_len_key = "distractor_{}".format(i)
        # Distractor Length feature
        dis_len = len(next(vocab._tokenizer([distractor])))
        example.features.feature[dis_len_key].int64_list.value.extend(dis_transformed)
    return example

def create_tfrecords_file(input_filename, output_filename, example_fn):
    """
    Creates a TFRecords file for the given input data and example transformation function
    """
    writer = tf.python_io.TFRecordsWriter(output_filename)
    print("Creating TFRecords file at {}...". format(output_filename))
    for i, row in enumerate(create_csv_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("Wrote to {}".format(output_filename))

