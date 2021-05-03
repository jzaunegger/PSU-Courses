# -*- coding:utf8 -*-
"""
This py page is for the Modeling and training part of this NLM. 
Try to edit the place labeled "# TODO"!!!
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, datetime, random
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
EPOCHS = 5

trigrams = []
vocab = {}

def word2index(word, vocab):
    """
    Convert an word token to an dictionary index
    """
    if word in vocab:
        value = vocab[word]
    else:
        value = -1
    return value

def index2word(index, vocab):
    """
    Convert an word index to a word token
    """
    for w, v in vocab.items():
        if v == index:
            return w
    return 0

def preprocess(file, is_filter=True):
    """
    Prepare the data and the vocab for the models.
    For expediency, the vocabulary will be all the words
    in the dataset (not split into training/test), so
    the assignment can avoid the OOV problem.
    """
    with open(file, 'r') as fr:
        for idx, line in enumerate(fr):
            words = word_tokenize(line)
            if is_filter:
                words = [w for w in words if not w in stop_words]
                words = [word.lower() for word in words if word.isalpha()]
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
            if len(words) > 0:
                for i in range(len(words) - 2):
                    trigrams.append(([words[i], words[i + 1]], words[i + 2]))
    print('{0} contain {1} lines'.format(file, idx + 1))
    print('The size of dictionary is：{}'.format(len(vocab)))
    print('The size of trigrams is：{}'.format(len(trigrams)))
    return 0


class NgramLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NgramLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 256)
        self.linear2 = nn.Linear(256, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def createRandomEmbedding(length, lowerBound, upperBound):
    embeddings = [0] * length

    for i in range(0, length):
        rand_val = random.uniform(lowerBound, upperBound)
        embeddings[i] = rand_val
    
    return embeddings

def output(model, file1, file2):
    """
    Write the embedding file to the disk
    """
    emb_min = 0
    emb_max = 0
    emb_length = 0
    with open(file1, 'w') as fw1:
        for word, id in vocab.items():

            word_data = [word2index(word, vocab)]
            word_tensor = torch.tensor(word_data, dtype=torch.long)
            word_emb = model.embeddings(word_tensor)

            emb_length = len(word_emb[0])

            emb_str = ''
            for i in range(0, len(word_emb[0])):
                emb_val = word_emb[0][i]
                current_emb_val = emb_val.item()
                emb_str += str(current_emb_val) + ' '

                if current_emb_val < emb_min:
                    emb_min = current_emb_val 

                if current_emb_val > emb_max:
                    emb_max = current_emb_val

            ostr = '{} {} \n'.format(word, emb_str)
            fw1.write(ostr)
        fw1.close()

    with open(file2, 'w') as fw2:
        for word,id in vocab.items():
            rand_emb = createRandomEmbedding(emb_length, emb_min, emb_max)
            embedding_string = ''
            for rand_entry in rand_emb:
                embedding_string += str(rand_entry) + ' '
            ostr = '{} {} \n'.format(word, embedding_string )
            fw2.write(ostr)
        fw2.close()

    print("------------------------------------------------------------------")
    print("Sucessfully Saved the embeddings to:")
    print(file1)
    print(file2)
    print("------------------------------------------------------------------")

def formatTime(seconds):
    time_format = str(datetime.timedelta(seconds=seconds))

    hrs = int(time_format[0:1])
    mins = int(time_format[2:4])
    secs = float(time_format[5:])
    secs_form = float(secs)

    if hrs == 1:
        if mins == 1:
            return '{} hour, {} minute, and {:.2f} seconds'.format(hrs, mins, secs)
        else:
            return '{} hour, {} minutes, and {:.2f} seconds'.format(hrs, mins, secs)

    else:
        if mins == 1:
            return '{} hours, {} minute, and {:.2f} seconds'.format(hrs, mins, secs)
        else:
            return '{} hours, {} minutes, and {:.2f} seconds'.format(hrs, mins, secs)

    return '{} hours, {} minutes, and {:.2f} seconds'.format(hrs, mins, secs_form)


def training():
    """
    Train the NLM
    """
    preprocess('./data/reviews_500.txt')
    losses = []

    # Create model, define loss function and optimizer
    model = NgramLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Record the start of the training time
    train_start = time.time()

    print(' ')
    print('Training Model (Creating Embeddings)')
    print("------------------------------------------------------------------")

    # Iterate through epochs
    for epoch in range(EPOCHS):
        total_loss = 0

        # Load in the dataset, and step through the batches
        dataL = torch.utils.data.DataLoader(trigrams, batch_size=64, shuffle=False)
        for context_dataL, target_dataL in dataL:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors

            context_word_1_arr = list(context_dataL[0])
            context_word_2_arr = list(context_dataL[1])
            target_dataL_arr = list(target_dataL)

            # Process the current batch data to determine the current trigrams
            current_trigrams = []
            for i in range(0, len(target_dataL_arr)):
                current_trigram_entry = []

                current_context = []
                current_context.append(context_word_1_arr[i])
                current_context.append(context_word_2_arr[i])

                current_trigram_entry = [current_context, target_dataL_arr[i]]
                current_trigrams.append(current_trigram_entry)

            # For each trigram
            for gram in current_trigrams:

                # Create context tensor
                contextData = [word2index(word, vocab) for word in gram[0]]
                context_tensor = torch.tensor(contextData, dtype=torch.long)

                target = gram[1]
                target_data = [word2index(target, vocab)]
                target_tensor = torch.tensor(target_data, dtype=torch.long)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                optimizer.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                probab = model.forward(context_tensor)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor) assert (0 == 1)
            loss = loss_function(probab, target_tensor)

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

            #print('Completed batch with a loss of {}'.format(float(loss.item())))

        losses.append(total_loss)
        print('Epoch {}, had a loss of {:.2f}'.format(epoch, float(loss.item())))

    # Record the end of training time
    train_end = time.time()
    time_diff = train_end - train_start
    time_str = formatTime(time_diff)

    print("------------------------------------------------------------------")
    print('Training Completed')
    print('Training Time: {}'.format(time_str))
    print('Final Loss: {:.2f}'.format(loss))
    #print('Total Losses at each epoch{} '.format(losses))  # The loss decreased every iteration over the training data!
    output(model, './data/embedding.txt', './data/random_embedding.txt')

if __name__ == '__main__':
    training()
