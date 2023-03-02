# Handwriting-synthesis-with-transformers

IMPORTANT ADJUSTMENTS:

* batch_first has been set to True in TransformerEncoderLayer to receive batches in the shape (batch, seq, feature) from the data loader
* adjusted mixture_weights to return weights for an entire sequence
* adjusted sample_uncond to use the entire sequence until the given point

TODO:
* compare loss to a simple baseline for generating mixture parameters
* calculate mdn loss sequence-wise instead of looping on strokes
* compare encoder-only, decoder-only and encoder-decoder for unconditional generation
* compare to RNN with attention
* visualize attention scores
* implement conditional generation: encoder receives text, decoder receives strokes
* verify if initial linear leayer is indeed needed; test different hidden sizes; see what representations are
* try embedding from space grid
