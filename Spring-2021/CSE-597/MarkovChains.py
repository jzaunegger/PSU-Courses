'''
    This application is about using Markov Chains to generate text using 
    N grams. A markov chain is essentially, a series of states, where each state 
    relates to one another in a logical fashion. In the case of text generation,
    a noun phrase is always followed by a verb phrase. 

        Lets say we have two states: A and B

            A -> B  with a 25% chance of occuring
            A -> B  with a 25% chance of occuring 
            B -> A  with a 25% chance of occuring
            B -> B  with a 25% chance of occuring

    This idea can be applied in a ton of different ways, from text generation to
    weather predictions or financial analysis. 

'''
import os, random

# Set parameters
input_path = os.path.join(os.getcwd(), 'Corpus', 'Lovecraft', 'Azathoth.txt') 
ngram_size = 5
markov_count = 1000
n_grams = {}
input_text = ''

# Read the file
with open(input_path, 'r') as txt_file:
    input_text = txt_file.read()


# Determine the N-Grams
for i in range(len(input_text)-ngram_size):
    gram = input_text[i:i+ngram_size]

    # Check that we have enough characters to get a next char
    if i == len(input_text) - ngram_size:
        break
    else:
        next_char = input_text[i+ngram_size]

    # Check if ngram is already in the dictionary
    if gram in n_grams.keys():
        pass
    else:
        n_grams[gram] = []

        # Append next character
    n_grams[gram].append(next_char)

# Generate new text from the ngram analysis
current_gram = input_text[0:ngram_size]
result = current_gram

for k in range(markov_count):
    possibilities = n_grams[current_gram]
    if len(possibilities) == 0:
        break

    next_char = random.choice(possibilities)
    result += next_char

    current_gram = result[len(result)-ngram_size:len(result)]
print(result)