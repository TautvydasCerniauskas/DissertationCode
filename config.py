import os

# Set up paths for dataset files
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
save_dir = os.path.join("data", "save")
# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

MIN_COUNT = 3  # Minimum word count threshold for trimming
MAX_LENGTH = 15  # Maximum sentence length to consider
small_batch_size = 5 # Batches used to feed into Encoder

# Configure models
model_name = 'cb_model'
# attn_model = 'dot'
# attn_model = 'general'
attn_model = 'concat'
hidden_size = 512  # Hidden size
encoder_n_layers = 2  # Encoder number of layers
decoder_n_layers = 2  # Encoder number of layers
dropout = 0.1  # Dropout value
batch_size = 64  # Batch size

# Configure training/optimization
clip = 50.0  # Gradient clipping value
teacher_forcing_ratio = 1.0  # Teacher forcing ratio
learning_rate = 0.0001  # Learning ratio value
decoder_learning_ratio = 5.0  # Decoder learning ratio
n_iteration = 50000  # Number of training iterations
print_every = 1  # How often the iteration will be printed out
save_every = 500  # Save iteration every N
checkpoint_iter = 50000  # Final iteration count, if this number is not met continue training
