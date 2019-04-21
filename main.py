import os
import torch
import torch.nn as nn
import argparse
from torch import optim

from load import voc, pairs
from config import *
from model import *
from train import trainIters
from evaluate import evaluateInput


def parse():

    parser = argparse.ArgumentParser(
        description="Seq2seq chatbot with Attention")
    parser.add_argument(
        '-tr',
        '--train',
        action='store_true',
        help="Train the model")
    parser.add_argument(
        '-eval',
        '--evaluate',
        action='store_true',
        help="Evaluate the model")
    parser.add_argument(
        '-b',
        '--beam',
        type=int,
        default=2,
        help='Beam size')
    parser.add_argument(
        '-n',
        '--name',
        default='{}'.format(model_name),
        help="Training name")

    args = parser.parse_args()
    return args


def run(args):

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model,
        embedding,
        hidden_size,
        voc.num_words,
        decoder_n_layers,
        dropout)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(),
        lr=learning_rate *
        decoder_learning_ratio)

    if(args.train):
        loadFilename = None
        print('Building encoder and decoder ...')
        print('Building optimizers ...')
        print('Models built and ready to go!')

        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()

        # Run training iterations
        print("Starting Training!")
        trainIters(
            model_name,
            voc,
            pairs,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            embedding,
            encoder_n_layers,
            decoder_n_layers,
            save_dir,
            n_iteration,
            batch_size,
            print_every,
            save_every,
            clip,
            corpus_name,
            loadFilename,
            args.name)

    if(args.evaluate):

        # Set checkpoint to load from; set to None if starting from scratch
        loadFilename = os.path.join(save_dir,
                                    model_name,
                                    corpus_name,
                                    '{}-{}_{}_{}'.format(encoder_n_layers,
                                                      decoder_n_layers,
                                                      hidden_size, args.name),
                                    '{}_checkpoint.tar'.format(checkpoint_iter))

        # Load model if a loadFilename is provided
        if loadFilename:
            # If loading on same machine the model was trained on
            # checkpoint = torch.load(loadFilename)
            # If loading a model trained on GPU to CPU
            checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            voc.__dict__ = checkpoint['voc_dict']

        if loadFilename:
            embedding.load_state_dict(embedding_sd)
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, voc, args.beam, args.name)


if __name__ == "__main__":
    args = parse()
    run(args)
