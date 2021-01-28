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