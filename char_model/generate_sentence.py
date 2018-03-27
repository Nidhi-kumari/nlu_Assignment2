from keras.models import model_from_json
import numpy as np


file = open('modelc1.json', 'r')
jmodel = file.read()
file.close()


model = model_from_json(jmodel)
model.load_weights("modelc.h5")



with open("input.txt", encoding='utf-8') as f:
    text = f.read().lower()

maxlen = 8

chars = sorted(list(set(text)))

char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))


def sample(preds, c=1.0):
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / c
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_sen(epoch):
    

    start_index = np.random.randint(0, len(text) - maxlen - 1)
    for value in [0.5]:
        
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        
        output = ''
        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char2index[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            
            next_index = sample(preds, value)
            
            next_char = index2char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            output = output + next_char
       
        print(output[:50])
        


generate_sen(100)
