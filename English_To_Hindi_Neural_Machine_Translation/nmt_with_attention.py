import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
import string
import random

tf.enable_eager_execution()

num_examples = 100
data = open('eng_to_hin.txt', 'r', encoding='utf8', errors='ignore').read()
lines = data.split('\n')
lines = lines[:num_examples]

english = [l.split('\t')[0] for l in lines]
hindi = [l.split('\t')[1] for l in lines]
hindi = ['<s> '+h+' <e>' for h in hindi]

english_vocab = sorted(list(set([w for e in english for w in e.split()])))
hindi_vocab = sorted(list(set([w for h in hindi for w in h.split()])))

english_vocab_size = len(english_vocab)
hindi_vocab_size = len(hindi_vocab)

max_english_len = max([len(e.split()) for e in english])
max_hindi_len = max([len(h.split()) for h in hindi])

english_w_to_i = {w: i for i, w in enumerate(english_vocab)}
hindi_w_to_i = {w: i for i, w in enumerate(hindi_vocab)}
english_i_to_w = {i: w for i, w in enumerate(english_vocab)}
hindi_i_to_w = {i: w for i, w in enumerate(hindi_vocab)}

input_tensor = numpy.zeros((len(english), max_english_len))
target_tensor = numpy.zeros((len(hindi), max_hindi_len))
for i in range(len(english)):
    for j in range(len(english[i].split())):
        input_tensor[i, j] = english_w_to_i[english[i].split()[j]]
for i in range(len(hindi)):
    for j in range(len(hindi[i].split())):
        target_tensor[i, j] = hindi_w_to_i[hindi[i].split()[j]]


shuffle_size = len(input_tensor)
batch_size = 64
embedding_size = 256
units = 1024

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(shuffle_size)
dataset = dataset.batch(batch_size, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_size, encoder_size, bt_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_size)
        self.encoder_size = encoder_size
        self.batch_size = bt_size
        self.gru = tf.keras.layers.GRU(self.encoder_size,  return_sequences=True, return_state=True)

    def call(self, x, h_state):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=h_state)
        return output, state

    def initial_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_size))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, emb_size, decoder_size, bt_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_size)
        self.decoder_size = decoder_size
        self.batch_size = bt_size
        self.gru = tf.keras.layers.GRU(self.decoder_size, return_sequences=True, return_state=True)

        self.fc = tf.layers.Dense(vocab_size)

        self.W1 = tf.keras.layers.Dense(self.decoder_size)
        self.W2 = tf.keras.layers.Dense(self.decoder_size)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, h_state, enc_output):
        hidden_state_with_time_axis = tf.expand_dims(h_state, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output)+self.W2(hidden_state_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights*enc_output, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

    def initial_hidden_state(self):
        return tf.zeros((self.batch_size, self.decoder_size))


encoder = Encoder(english_vocab_size, embedding_size, units, batch_size)
decoder = Decoder(hindi_vocab_size, embedding_size, units, batch_size)

optimizer = tf.train.AdamOptimizer()


def loss_function(real, prediction):
    mask = 1-numpy.equal(real, 0)
    # loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=prediction)*mask
    loss_val = tf.keras.losses.sparse_categorical_crossentropy(real, prediction)*mask
    return tf.reduce_mean(loss_val)


checkpoint_dir = 'checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)

epochs = 100

for epoch in range(epochs):
    start_time = time.time()
    print('Epoch = {} Start...'.format(epoch+1))
    hidden_state = encoder.initial_hidden_state()
    total_loss = 0
    for batch, (input_seq, target_seq) in enumerate(dataset):
        loss = 0
        with tf.GradientTape() as tape:
            encoder_output, encoder_hidden_state = encoder(input_seq, hidden_state)
            decoder_hidden_state = encoder_hidden_state
            decoder_input = tf.expand_dims([hindi_w_to_i['<s>']]*batch_size, 1)
            for t in range(1, target_seq.shape[1]):
                predictions, decoder_hidden_state, _ = decoder(decoder_input, decoder_hidden_state, encoder_output)
                loss += loss_function(target_seq[:, t], predictions)
                decoder_input = tf.expand_dims(target_seq[:, t], 1)
        batch_loss = loss/int(target_seq.shape[1])
        total_loss += batch_loss
        variables = encoder.variables+decoder.variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    checkpoint.save(file_prefix=checkpoint_prefix)
    end_time = time.time()
    print('Epoch = {}\tEnd.\nTime Taken = {:.2f} secs\n'.format(epoch+1, end_time-start_time))


def encode_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.translate(str.maketrans('', '', string.digits))
    sentence = ' '.join(sentence.split())
    encoded_sentence = numpy.zeros((1, max_english_len))
    for i in range(len(sentence.split())):
        encoded_sentence[:, i] = english_w_to_i[sentence.split()[i]]
    return encoded_sentence


def evaluate(input_sentence):
    attention = numpy.zeros((max_hindi_len, max_english_len))
    encoded_input_sentence = encode_sentence(input_sentence)
    encoded_input_sentence = tf.convert_to_tensor(encoded_input_sentence)
    predicted_sentence = ''
    enc_hidden_state = tf.zeros((1, units))
    enc_output, enc_hidden_state = encoder(encoded_input_sentence, enc_hidden_state)
    dec_hidden_state = enc_hidden_state
    dec_input = tf.expand_dims([hindi_w_to_i['<s>']], 0)
    for i in range(max_hindi_len):
        predicted_output, dec_hidden_state, attention_weights = decoder(dec_input, dec_hidden_state, enc_output)
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention[i] = attention_weights.numpy()
        predicted_index = tf.argmax(predicted_output[0]).numpy()
        predicted_sentence += hindi_i_to_w[predicted_index]+' '
        if hindi_i_to_w[predicted_index] == '<e>':
            return input_sentence, predicted_sentence, attention
        dec_input = tf.expand_dims([predicted_index], 0)
    return input_sentence, predicted_sentence, attention


def plot_attention(input_sentence, predicted_sentence, attention):
    figure = plt.figure(figsize=(10, 10))
    axis = figure.add_subplot(1, 1, 1)
    axis.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    axis.set_xticklabels([''] + input_sentence, fontdict=fontdict, rotation=90)
    axis.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    plt.show()


def translate(sentence):
    input_sentence, predicted_sentence, attention = evaluate(sentence)
    print('Input: {}'.format(input_sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
    attention = attention[:len(predicted_sentence.split(' ')), :len(input_sentence.split(' '))]
    # plot_attention(input_sentence.split(), predicted_sentence.split(), attention)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# translate('your sentence')
