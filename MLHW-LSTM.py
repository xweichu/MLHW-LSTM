import csv
import numpy as np
import tensorflow as tf
import re
import datetime
#from random import randint
import random
import math
import csv
#test github

def data_process(filename):
    raw_data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp = [row[0]]
            line = row[1].split('https')
            temp.append(line[0])
            raw_data.append(temp)
        raw_data.pop(0)
    return raw_data

def loadGloveModel(gloveFile):
    f = open(gloveFile, 'r')
    wordlist = []
    vectors = []
    for line in f:
        splitLine = line.split()
        wordlist.append(splitLine[0])
        vectors.append([float(val) for val in splitLine[1:]])
    vectors = np.array(vectors)
    return wordlist, vectors

def get_idmatrix(train_data, word_list, seqlen):
    ids = np.zeros((len(train_data), seqlen), dtype='int32')
    i = 0
    for sentence in train_data:
        split = sentence[1].split()
        j = 0 
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        for word in split:
            word = re.sub(strip_special_chars, "", word.lower())
            try:
                ids[i][j] = word_list.index(word)
            except ValueError:
                ids[i][j] = 410055
            j = j + 1
        i = i + 1
    return ids

def getTrainBatch(train_data, ids, batchSize, maxSeqLength):
    labels = []
    batch_seqlen = []
    arr = np.zeros([batchSize, maxSeqLength])
    rd = random.sample(range(len(train_data)), batchSize)
    for i in range(batchSize):
        num = rd[i]
        if train_data[num][0] == 'realDonaldTrump':
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num]
        n = np.count_nonzero(arr[i])
        batch_seqlen.append( n - 1 if n > 0 else 0)
    return arr, labels, batch_seqlen


batchSize = 189
lstmUnits = 256
layer_num = 4 
numClasses = 2
maxSeqLength = 150
numDimensions = 50
iterations = 201

train_data = data_process('train.csv')
word_list, word_vectors = loadGloveModel("glove.twitter.27B.50d.txt")
#ids = get_idmatrix(train_data,word_list, maxSeqLength)
#np.save('ids',ids)
ids = np.load('ids.npy')

# training................................................................................
tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors, input_data)
data = tf.cast(data, tf.float32)

stacked_rnn = []
keep_prob = tf.placeholder(tf.float32)
for i in range(layer_num):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
    stacked_rnn.append(lstm_cell)
mlstm_cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

init_state = mlstm_cell.zero_state(batchSize, dtype=tf.float32)

value, _ = tf.nn.dynamic_rnn(mlstm_cell, inputs=data, initial_state=init_state, time_major=False)
value = tf.nn.dropout(value, keep_prob)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), dtype=tf.float32)

seqlen = tf.placeholder(tf.int32, [None])
index = tf.range(0, batchSize) * maxSeqLength + seqlen
# Indexing
last= tf.gather(tf.reshape(value, [-1, lstmUnits]), index)
#last = tf.gather(value, idx)
last = tf.cast(last, tf.float32)
#bias = tf.cast(bias, tf.float32)
#weight = tf.cast(weight, tf.float32)
logits = tf.matmul(last, weight) + bias
prediction = tf.nn.softmax(logits)

#test
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

#loss = - tf.reduce_mean(labels * tf.log(prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels, nextseqlen = getTrainBatch(train_data,ids,batchSize,maxSeqLength);
   lo, op = sess.run([loss, optimizer], {input_data: nextBatch, labels: nextBatchLabels, keep_prob: 0.5, seqlen: nextseqlen})
   print('>>>batch train, which iteration = %d' % (i))
   #Write summary to Tensorboard
   if (i % 50 == 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels, keep_prob: 0.5, seqlen: nextseqlen})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
writer.close()

def getTestBatch(test_ids, batchSize, cursor):
    batch_seqlen = []
    vec = test_ids[cursor: cursor + batchSize]
    batch_seqlen = np.count_nonzero(vec, axis=1)
    return vec, batch_seqlen
    


test_data = data_process('test.csv')
test_ids = get_idmatrix(test_data,word_list, maxSeqLength)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

with open('pred.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'realDonaldTrump', 'HillaryClinton'])
    for i in range(int(len(test_data) / batchSize)):
        #nextBatch = test_ids[i:i + batchSize]
        nextBatch, seq = getTestBatch(test_ids, batchSize, i * batchSize)
        # nextBatchLabels = np.array
        pred = sess.run(prediction, {input_data: nextBatch, keep_prob:0.5, seqlen: seq})
        for idx in range(np.shape(pred)[0]):
            spamwriter.writerow([i * batchSize + idx, "%.6f" % (pred[idx][0]), "%.6f" % (pred[idx][1])])
