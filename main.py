import os
import torch
import torch.nn as nn
import argparse
from torch import optim

from load import loadPrepareData
from config import *
from model import *
from train import trainIters
from evaluate import evaluateInput


# # Load/Assemble voc and pairs
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# # Print some pairs to validate
# print("\npairs:")
# for pair in pairs[:10]:
#     print(pair)

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 50000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

# print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
# print('Models built and ready to go!')

# # Ensure dropout layers are in train mode
# encoder.train()
# decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

def parse():
    parser = argparse.ArgumentParser(description="Seq2seq chatbot with Attention")
    parser.add_argument('-tr', '--train', action='store_true', help="Train the model")
    parser.add_argument('-eval', '--evaluate', action='store_true', help="Evaluate the model")
    parser.add_argument('-b', '--beam', type=int, default=1, help='Beam size')


    args = parser.parse_args()
    return args

def run(args):
    if(args.train):
        # Run training iterations
        print("Starting Training!")
        trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, loadFilename)
    if(args.evaluate):
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, voc, args.beam)

if __name__ == "__main__":
    args = parse()
    run(args)