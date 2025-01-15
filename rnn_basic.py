from rnn_basic_data import train_data, test_data
import numpy as np
import random

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
# print(vocab_size)  # 18

# assign a index to each word: creating a vector

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
# print(idx_to_word)

# one hot-encoding
def createInputs(text):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1;
        inputs.append(v)
    return inputs


# forward phase
class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):

        self.Whh = np.random.randn(hidden_size, hidden_size)/1000
        self.Wxh = np.random.randn(hidden_size, input_size)/1000
        self.Why = np.random.randn(output_size, hidden_size)/1000

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}


        # Perform each step of the RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh@x + self.Whh@h + self.bh)
            self.last_hs[i+1] = h

        y = self.Why@h + self.by

        return y, h
    
    def backprop(self, dy, lr=0.01):
        n = len(self.last_inputs)

        d_Why = dy@self.last_hs[n].T
        d_by = dy

        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.Why.T@dy

        # BPTT
        for t in reversed(range(n)):
            temp = ((1-self.last_hs[t+1]**2)*d_h)

            d_bh += temp
            d_Whh += temp@self.last_hs[t].T
            d_Wxh += temp@self.last_inputs[t].T

            d_h = self.Whh@temp

            # To prevent the vanishing gradient
            for d in [d_Wxh, d_Why, d_bh, d_by]:
                np.clip(d, -1, 1, out=d)

            self.Whh -= lr*d_Whh
            self.Wxh -= lr*d_Wxh
            self.Why -= lr*d_Why
            self.bh -= lr*d_bh
            self.by -= lr*d_by


        
    
def softmax(xs):
    return np.exp(xs)/sum(np.exp(xs))


rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)

def processData(data, backprop=True):
    items = list(data.items())
    # print('items: ', items)
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)
        # print(f'inputs: {inputs}, target: {target}')

        out, _ = rnn.forward(inputs)
        probs = softmax(out)
        # print(f'out: {out}, probs: {probs}')

        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            rnn.backprop(d_L_d_y)

    return loss/len(data), num_correct/len(data)


for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))


