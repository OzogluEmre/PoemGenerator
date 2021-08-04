import json
import dynet as dy
import numpy as np
from numpy.random import choice
from math import log, pow

# here we create bigram corpus and add tokens to word dictionary
def bigram_tokenizer(poem, dataset, add_dict=True):
    poem = str(poem).replace("\n", " \n ")

    # start and end token
    poem = "bos {} eos".format(poem)
    tokens = poem.split(" ")

    if add_dict:
        for token in tokens:
            if token in token_dict:
                token_dict[token] += 1
            else:
                token_dict[token] = 1

    ngrams = zip(*[tokens[i:] for i in range(2)])
    for ngram in ngrams:
        dataset.append(ngram)


# calculates perplexity with given probability list
def perplexity(_probability, count):
    return pow(2, ((-1) / count) * _probability)

# here we calculate our input vector from the glove embeddings also OOV words from our poem corpus will be created as a new vector in the glove embeddings
def glove_vector(pair):
    if pair[0]=="\n":
        nline_array=np.random.random(GLOVE_SIZE)
        embeddings_dict.update({'\n': nline_array})
        input = embeddings_dict.get(pair[0])
        
    elif pair[0] not in embeddings_dict.keys():
        oov_array = np.random.random(GLOVE_SIZE)
        embeddings_dict.update({'OOV': oov_array})
        input = embeddings_dict.get('OOV')
        
    else:
        input=embeddings_dict.get(pair[0])

    return input


if __name__ == "__main__":

    bigram_dataset = []
    token_dict = dict()

    # TASK 1 BEGINS HERE

    with open("unim_poem.json", "r") as f:
        poem_json = json.load(f)

    # here we create our bigram dataset
    for _poem in poem_json[:100]:
        bigram_tokenizer(_poem["poem"], bigram_dataset, add_dict=True)


    # this is the index to word list
    index2word = list(token_dict.keys())

    # this is the word to index dictionary
    word2index = dict()
    for index, token in enumerate(index2word):
        word2index[token] = index


    # Here we acquire our glove embeddings
    embeddings_dict = {}
    with open("glove.6B.200d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


    # Here we set our models parameters
    INPUT_SIZE = len(word2index.keys())
    OUTPUT_SIZE = len(word2index.keys())
    HIDDEN_SIZE = 10
    GLOVE_SIZE = len(embeddings_dict.get("start"))

    model = dy.ParameterCollection()
    pH = model.add_parameters((HIDDEN_SIZE, GLOVE_SIZE))
    pb = model.add_parameters(INPUT_SIZE)
    pU = model.add_parameters((OUTPUT_SIZE, HIDDEN_SIZE))
    pd = model.add_parameters(HIDDEN_SIZE)

    #here we choose our model such as SGDT,ADAM or etc
    trainer = dy.SimpleSGDTrainer(model)

    EPOCHS = 10

    print("Training has begin.")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for pair in bigram_dataset:

            #here we add parameters
            input_vector = glove_vector(pair)
            output_vector=word2index.get(pair[1])
            dy.renew_cg()
            H = dy.parameter(pH)
            U = dy.parameter(pU)
            b = dy.parameter(pb)
            d = dy.parameter(pd)
            x = dy.inputVector(input_vector)


            # here we get prediction
            y = U * dy.tanh(H * x + d) + b

            # here we calculate loss
            loss = dy.pickneglogsoftmax(y, output_vector)

            epoch_loss += loss.scalar_value()

            loss.backward()
            trainer.update()

        print("Epoch %d. loss = %f" % (epoch, epoch_loss / len(bigram_dataset)))

    #here we save the model
    model.save("train_.model")

    ########HERE TASK 2 BEGINS

    #Here the number of lines will be taken from the user and in each return of the loop , i set it up for 5 poems here
    for i in range(5):

        newline = 0

        # poem starts with start token which is "bos" for and ends with end token which is "eos" according to Glove embeddings
        start = "bos"
        end = "eos"
        line = start
        poem = ""
        total_probability = 0
        length = input("Please enter the number of lines: ")
        while newline < int(length):

            #here we set parameters
            input_vector = glove_vector(pair)
            dy.renew_cg()
            H = dy.parameter(pH)
            U = dy.parameter(pU)
            b = dy.parameter(pb)
            d = dy.parameter(pd)
            x = dy.inputVector(input_vector)

            # here we get prediction
            y = U * dy.tanh(H * x + d) + b

            # this process normalize the probability list between 0 and 1.
            probabilities = list(np.exp(y.npvalue()) / sum(np.exp(y.npvalue())))

            # weighted choice function. picks random from unique word list
            predicted = choice(index2word, p=probabilities)

            # word probability for perplexity
            word_probability = probabilities[word2index[predicted]]

            if predicted == "\n":
                # if predicted and start token is newline, continue
                if start == "\n":
                    continue
                poem += "{}\n".format(line.replace("bos ", "").replace("eos", ""))
                line = ""
                newline += 1

            if start == "\n":
                line = predicted
            else:
                line = "{} {}".format(line,predicted)

            start = predicted

            # this is used to for calculation for perplexity
            total_probability += log(word_probability, 2)
        print("\nPOEM #{}\n".format(i + 1))
        print(poem)

        # TASK 3 BEGINS
        # this will used for perplexity
        poem_bigram_corpus = []
        bigram_tokenizer(poem, poem_bigram_corpus, add_dict=False)

        # here we calculate perplexity
        poem_perplexity = perplexity(total_probability, len(poem_bigram_corpus))
        print("Poem-{} Perplexity: {}\n".format(i + 1, poem_perplexity))
