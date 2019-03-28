import os

# Set up paths for dataset files
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
save_dir = os.path.join("data", "save")
# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

MAX_LENGTH = 15  # Maximum sentence length to consider
small_batch_size = 5

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 50000
print_every = 1
save_every = 500