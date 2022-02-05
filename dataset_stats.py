# loading data
data = open('eng_to_hin.txt', 'r', encoding='utf8', errors='ignore').read()
lines = data.split('\n')
# lines = lines[:1000]
n_data = len(lines)

print('No of Pairs = ', n_data)

# separating english and hindi text
english = [l.split('\t')[0] for l in lines]
hindi = [l.split('\t')[1] for l in lines]

# determining max sequence length
max_english_seq_len = max([len(e) for e in english])
max_hindi_seq_len = max([len(h) for h in hindi])

print('Maximum English Sequence Length = ', max_english_seq_len)
print('Maximum Hindi Sequence Length = ', max_hindi_seq_len)

# unique words
english_words = [w for e in english for w in e.split()]
hindi_words = [w for h in hindi for w in h.split()]

english_words = sorted(list(set(english_words)))
hindi_words = sorted(list(set(hindi_words)))
n_english_words = len(english_words)
n_hindi_words = len(hindi_words)

print('No. Of Unique English Words = ', n_english_words)
print('No. Of Unique Hindi Words = ', n_hindi_words)

# unique characters
english_chars = [c for e in english for c in e]
hindi_chars = [c for h in hindi for c in h]

english_chars = sorted(list(set(english_chars)))
hindi_chars = sorted(list(set(hindi_chars)))
n_english_chars = len(english_chars)
n_hindi_chars = len(hindi_chars)

print('No. Of Unique English Characters = ', n_english_chars)
print('No. Of Unique Hindi Characters = ', n_hindi_chars)

# mapping words to numbers
eng_w_to_i = {w: i for i, w in enumerate(english_words)}
hin_w_to_i = {w: i for i, w in enumerate(hindi_words)}

# reverse mapping
eng_i_to_w = {i: w for i, w in enumerate(english_words)}
hin_i_to_w = {i: w for i, w in enumerate(hindi_words)}

# unique words with frequency -> write to file
english_words_freq = {w: 0 for w in english_words}
hindi_words_freq = {w: 0 for w in hindi_words}

for e in english:
    for w in e.split():
        english_words_freq[w] += english_words_freq.get(w) + 1

for h in hindi:
    for w in h.split():
        hindi_words_freq[w] += hindi_words_freq.get(w) + 1

file_eng_words_freq = open('english_words_freq.txt', 'w', encoding='utf8', errors='ignore')
for w in english_words_freq:
    file_eng_words_freq.write('{}\t{}\n'.format(w, english_words_freq[w]))

file_hin_words_freq = open('hindi_words_freq.txt', 'w', encoding='utf8', errors='ignore')
for w in hindi_words_freq:
    file_hin_words_freq.write('{}\t{}\n'.format(w, hindi_words_freq[w]))

# unique characters with frequency -> write to file
english_chars_freq = {c: 0 for c in english_chars}
hindi_chars_freq = {c: 0 for c in hindi_chars}

for e in english:
    for c in e:
        english_chars_freq[c] += english_chars_freq.get(c) + 1

for h in hindi:
    for c in h:
        hindi_chars_freq[c] += hindi_chars_freq.get(c) + 1

file_eng_chars_freq = open('english_chars_freq.txt', 'w', encoding='utf8', errors='ignore')
for c in english_chars_freq:
    file_eng_chars_freq.write('{}\t{}\n'.format(c, english_chars_freq[c]))

file_hin_chars_freq = open('hindi_chars_freq.txt', 'w', encoding='utf8', errors='ignore')
for c in hindi_chars_freq:
    file_hin_chars_freq.write('{}\t{}\n'.format(c, hindi_chars_freq[c]))
