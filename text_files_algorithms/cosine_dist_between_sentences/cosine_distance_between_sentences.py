import numpy as np
import re
import matplotlib.pyplot as plt
import scipy as sp
from scipy.spatial import distance

print("""
This script takes a .txt file of sentences
and measures how similar every of them
to the first sentence.
Similarity measure is cosine distance.

The algorithm's output: the index of the closest sentence
and its cosine distance to the first sentence
""")


def line_formatting(line):
    """returns lowercase formatted line from txt file with empty entries removed"""
    line = str.lower(line)
    line_splitted = re.split('[^a-z]', line)
    line_cleaned = [x for x in line_splitted if x != '']
    return line_cleaned


def cos_dict_to_sentence(word_freq_matrix, sentence_numb_from, sentence_numb_to):
    """Calculates cosine distance between two sentences"""
    cos_dist = sp.spatial.distance.cosine(word_freq_matrix[sentence_numb_from], word_freq_matrix[sentence_numb_to])
    return cos_dist


# Pass the name of the file
print("== Preparation: =="
      "\n1. Create a .txt file: each line is a new sentence"
      "\n2. place the .txt file into the folder with this script"
      "\nThen load the .txt file.")


while True:
    txt_file = input("== Your input == \nType here file_name+extension: ")
    if txt_file[-4:] == ".txt":
        if len(txt_file) > 0:
            print("Ok, File is added")
            break
        else:
            print("Error: No file name was passed")
    else:
        print(f"Error: File name '{input}' does not have extension '.txt'")

words_by_sentence = {}
all_words = []

with open(txt_file, 'r') as file_obj:
    for sentence, line in enumerate(file_obj):
        print(f'sentence #{sentence} is processed')
        line_cleaned = line_formatting(line)
        words_by_sentence[sentence] = line_cleaned
        for word in line_cleaned:
            if word not in all_words:
                all_words.append(word)
print(f'{len(all_words)} words found in the file {txt_file}')

# create a matrix of word frequency in each sentence
word_freq_in_sentences = np.zeros((len(words_by_sentence.keys()), len(all_words)))

for sentence in words_by_sentence.keys():
    words_in_sent = words_by_sentence[sentence]
    words, counts = np.unique(words_in_sent, return_counts=True)
    for word, count in zip(words, counts):
        indx = all_words.index(word)
        word_freq_in_sentences[sentence, indx] = int(count)

fig = plt.figure()
plt.imshow(word_freq_in_sentences)
plt.title("Heatmap of words frequency in sentences")
plt.xlabel("Word ID")
plt.ylabel("Sentence ID")
plt.colorbar()
plt.show()

cos_dist_list = []
from_sentence = 0
for to_sentence in words_by_sentence.keys():
    cos_dist = cos_dict_to_sentence(word_freq_in_sentences, from_sentence, to_sentence)
    val = (from_sentence, to_sentence, cos_dist)
    cos_dist_list.append(val)

cos_dist_list.sort(key=lambda x: x[2])
(from_sentence, to_sentence, cos_dist) = cos_dist_list[1]
print(f"The closest sentence to {from_sentence} is sentence # {to_sentence} with cos distance = {cos_dist}")
