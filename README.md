# SentimentRNN
A recurrent neural network that uses word embeddings to do sentiment analysis in both French and English. It uses Google's word2vec implementation written in C through the very good `word2vec` module.

## Creating word embeddings

From a terminal, enter `python3 run.py --data`

## Training

From a terminal, do `python3 train.py --embedding data/en.bin --model model/sentiment_rnn.pb`. The code will train a new neural network using the specified embedding.

## Testing

From a terminal, do `python3 test.py --model model/sentiment_rnn.pb --text test.txt`. The text example will be evaluated line-by-line to simply add \\n wherever you want to split it.
