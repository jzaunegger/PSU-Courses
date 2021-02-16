############################################################
# CSE 597: Homework 1
############################################################

student_name = "Jackson Zaunegger"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import re, random, math

############################################################
# Section 1: Markov Models
############################################################

# Split text into tokens, and return them as a list
def tokenize(text):
    # Use regular expressions to split by whitespace, and punctuation.
    # This method also protects unicode characters
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

# Determine a list of N-Grams from a token list.
def ngrams(n, tokens):

    #Initalize Variables
    n_grams = []
 
    # Append Start Tags   
    for i in range(0, n-1):
        tokens.insert(0, '<START>')

    # Add end tag
    tokens.append('<END>')

    # Split tokens into N-Grams
    for j in range(0, len(tokens)-(n-1)):
        gram = tokens[j:j+n]
        gram_len = len(gram)
        new_gram = ( tuple(gram[0:n-1]), gram[gram_len-1] )
        n_grams.append(new_gram)

    # Return the ngram list
    return n_grams

# Class for the NgramModel 
class NgramModel(object):

    # Constructor Function
    def __init__(self, n):

        # Set input variables
        self.n = n

        # Initalize variables
        self.ngrams = {}
        self.total_grams = 0

    # Take a new sentence, tokenize it, then convert those tokens to ngrams.
    # Then take the ngrams, and add them to the dictionary and update the counts.
    def update(self, sentence):

        # Convert sentence to ngrams
        tokens = tokenize(sentence)
        grams = ngrams(self.n, tokens)


        # Add ngrams to the ngram dictionary, and update the counts
        for gram in grams:

            context = gram[0]
            token = gram[1]

            # Check if context is in the list
            if context in self.ngrams.keys():

                current_tokens = (self.ngrams[context]['tokens'])
                
                # If the token is in the tokens list, update the count
                if token in current_tokens:
                    token_index = current_tokens.index(token)
                    self.ngrams[context]['counts'][token_index] += 1
                
                # Add new token and counter
                else:
                    self.ngrams[context]['tokens'].append(token)
                    self.ngrams[context]['counts'].append(1)

            # Add new context to the list
            else:
                self.ngrams[context] = {"tokens": [token], "counts": [1]}

            self.total_grams += 1


    # Take a context and a token, and use the total number of ngrams to
    # calculate a probability. If the context/token pair, does not exist
    # it returns a 0.
    def prob(self, context, token):
        
        # Check if context exists
        if context in self.ngrams.keys():
            current_tokens = self.ngrams[context]['tokens']

            # Check if token exists in that context
            if token in current_tokens:

                # Determine the total counts for these tokens
                total = 0
                for count in self.ngrams[context]['counts']:
                    total += count

                token_index = current_tokens.index(token)
                token_count = self.ngrams[context]['counts'][token_index]
                return token_count / total
            
            # If token does not exist in that context, return 0
            else:
                return 0.0

        # Return 0
        else:
            return 0.0
            
    # Given a specific context, return a token
    def random_token(self, context):

        # If the context is valid
        if context in self.ngrams.keys():

            # Generate random value
            rand_val = random.random()

            # Create a dictionary of probabilities
            prob_dict = {}
            for token in self.ngrams[context]['tokens']:
                prob_dict[token] = self.prob(context, token)

            # Determine which token to return
            prob_sum = 0
            for token in sorted(prob_dict):
                prob_sum += prob_dict[token]
                if prob_sum > rand_val:
                    return token

        # Else, throw error
        else:
            print("Error: The given context could not be found.")

    # Given a number, generate that number of tokens to create
    # text, based on the training text(s).
    def random_text(self, token_count):

        # Determine the starting context
        num_starts = self.n - 1
        start_context = []

        for i in range(0, num_starts):
            start_context.append('<START>')

        # Set the starting context to the context list, and create a list
        # to keep track of the generated tokens
        contexts = start_context.copy()
        gen_tokens = []

        for i in range(0, token_count):

            # Determine new token, and add it to the list
            recent_context = tuple(contexts)
            current_token = self.random_token(recent_context)

            # Reset to the starting context
            if current_token =="<END>":
                gen_tokens.append(current_token)
                contexts = start_context.copy()

            else:
                gen_tokens.append(current_token)
                contexts.append(current_token)
                contexts.pop(0)

        # Stringify the generated tokens
        gen_text = ""
        for token in gen_tokens:
            gen_text += token + " "

        # Display the generated text
        print(gen_text)
        return gen_text

    # Take in a sentence, and calculate the perplexity
    def perplexity(self, sentence):
        sentence_tokens = tokenize(sentence)
        sentence_ngrams = ngrams(self.n, sentence_tokens)

        # First find the log equivalent to the inside of the root, 
        # from the perplexity equation
        total = 0
        for gram in sentence_ngrams:
            prob = self.prob(gram[0], gram[1])
            total += math.log(1 / prob)
        res = math.exp(total)

        # Take the root of the result
        perplexity = math.pow(res, 1/len(sentence_ngrams))
        return perplexity

# Create a model 
def create_ngram_model(n, path):

    # Read in the file
    with open(path, 'r') as txt_file:
        sentences = txt_file.readlines()

    # Create the model
    model = NgramModel(n)

    for i in range(0, len(sentences)):
        model.update(sentences[i])

    # Add the sentences to the model
    for sentence in sentences:
        model.update(sentence)

    return model

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
    About 5 to 6 hours.
"""

feedback_question_2 = """
    I found calculating the probabilites and the perplexity to be the
    most challenging. I was struggling to wrap my head around finding the
    best way to store the necessary data, and using it to calculate the 
    probabilities. I also was struggling with using log space to convert the perlexity,
    as using logs in this way is pretty new to me. Math is not my forte, so understanding
    how to break down formulas into implementation is always a bit of a struggle for me.
"""

feedback_question_3 = """
    I like this assignment as a whole, I think text generation and creative
    coding in general is extremely fun, and I am excited to see how I can use 
    this project and apply it to other writing styles, or further developing 
    this concept. One of the ways I can see to further develop this is to save
    the list of ngrams to a file and reload it, and continue to add more texts to
    it, to develop more generative possibilites. One such example, would be training 
    this system on texts from HP Lovecraft, one of my favorite authors. I would have
    maybe changed the notation of the perplexity equation to be a bit easier to 
    solve. (But it may just be me who struggled with this.)
"""