# SentimentRNN
A recurrent neural network that uses word embeddings to do sentiment analysis in both French and English. It uses Google's word2vec implementation written in C through the very good `word2vec` module.

## Creating word embeddings (word2vec.py folder)

From a terminal, enter `python3 word2vec.py embedding.npy wordlist.txt -i [YOUR_CORPUS] -vs 50000 -es 128 -bs 64`

## Training

From a terminal, do `python3 run.py --train -em embedding.npy -wl wordlist.txt`. The code will train a new neural network using the specified embedding.

## Testing

From a terminal, do `python3 run.py --test -m model/sentiment_rnn.pb --text test.txt`. The text example will be evaluated line-by-line to simply add \\n wherever you want to split it.
