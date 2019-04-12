from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import csv
import codecs
import pandas as pd
import numpy as np
import re

from config import datafile, corpus, corpus_name

# Print first 10 lines of a passed on file


def printLines(fileName, n=10):
    with open(fileName, 'rb') as datafile:
        lines = datafile.readlines()
        for line in lines[:n]:
            print(line)


printLines(os.path.join(corpus, "movie_lines.txt"))

printLines(os.path.join(corpus, "movie_conversations.txt"))

# Splits each line of the file into a dictionary of fields(lineID,
# characterID, movieID, character, text)


def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            # Split on +++$+++
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

# Groups fields of lines from `loadLines` into conversations based on
# *movie_conversations.txt*


def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            # Same thing as before, split files at +++$+++
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485',
            # 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                # Takes lines from the loadLines and appends them all together
                convObj["lines"].append(lines[lineId])
            # Creates a full conversation list with all the utterances,
            # characterIDs and so on.
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(
                len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            # Then we take the first line of a conversation as an input and a
            # second line as a target
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty we ignore them
            # and move to a another one)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


friends_df = pd.read_csv("data/friends_final_rdy.csv")
friends_sentences = friends_df['line']

# This is not a perfect implementation....


def extractSentencePairsFromCsv(sentences):
    qa_pairs = []
    for count, line in enumerate(sentences):
        if count % 2 == 0:
            inputLine = line.strip()
        else:
            targetLine = line.strip()
            qa_pairs.append([inputLine, targetLine])
    return qa_pairs


friends_pairs = extractSentencePairsFromCsv(friends_sentences)

# This is not a perfect implementation....


def extractSentencesFromHYMYMFile(filename):
    sentences_df = pd.read_csv(filename)
    sentences_df.dropna(subset=['Sentence'], inplace=True)

    sentences = []
    for i, line in enumerate(sentences_df['Sentence']):
        line = line.strip()
        line = re.sub(r'[\(\[].*?[\)\]]', '', line)
        line = line.split(":")[1:]
        line = ' '.join(line)
        sentences.append(line)
    sentences_df["Formatted"] = sentences
    sentences_df['Formatted'].replace('', np.nan, inplace=True)
    sentences_df.dropna(subset=['Formatted'], inplace=True)
    return sentences_df['Formatted']


HIMYM_sentences = extractSentencesFromHYMYMFile("data/HIMYM_sentences.csv")
HIMYM_pairs = extractSentencePairsFromCsv(HIMYM_sentences)

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
# Define our desired names for line fields
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
# Definied conversations fields
MOVIE_CONVERSATIONS_FIELDS = [
    "character1ID",
    "character2ID",
    "movieID",
    "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(
    os.path.join(corpus, "movie_conversations.txt"),
    lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file, where each input
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)
    for pair in friends_pairs:
        writer.writerow(pair)
    for pair in HIMYM_pairs:
        writer.writerow(pair)


# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)
print(len(datafile))
