import csv
import numpy as np
import tensorflow as tf
import re
import datetime
from random import randint
#import math
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
    f = open(gloveFile, 'rb')
    wordlist = []
    vectors = []
    for line in f:
        splitLine = line.split()
        wordlist.append(splitLine[0])
        vectors.append([float(val) for val in splitLine[1:]])
    vectors = np.array(vectors)
    return wordlist, vectors


def get_idmatrix(train_data, word_list):
    ids = np.zeros((len(train_data), 150), dtype='int32')
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

#change add a variable to record length of sequence
def getTrainBatch(train_data, ids, batchSize, maxSeqLength):
    labels = []
    batch_seqlen = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(1, len(train_data) - 1)
        if train_data[num][0] == 'realDonaldTrump':
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num - 1:num]
        batch_seqlen[i] = np.count_nonzero(arr[i])
    return arr, labels,batch_seqlen


train_data = data_process('train.csv')
word_list, word_vectors = loadGloveModel("glove.twitter.27B.50d.txt")
# ids = get_idmatrix(train_data,word_list)
# np.save('ids',ids)
ids = np.load('ids.npy')

#learning rate
batchSize = 27
lstmUnits = 64
numClasses = 2
maxSeqLength = 150
numDimensions = 50
iterations = 20

# training................................................................................
tf.reset_default_graph()
labels = tf.placeholder(tf.float64, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
# A placeholder for indicating each sequence length change
seqlen = tf.placeholder(tf.int32, [None])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
#lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.50)
value, states = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float64)# we should not use dynamic
# we should not use dynamic, since we have already input zero to make them have same sequence
#way 1. Is there any difference when using contrib rnn and nn
# value, states = tf.nn.static_rnn(lstmCell, data, dtype=tf.float64)
#way 2.
#value, states = tf.contrib.rnn.static_rnn(lstmCell, data, dtype=tf.float64)
#state??? as the input of logistic regression  change

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
#change--------------------------------
# Hack to build the indexing and retrieve the right output.
batch_size = tf.shape(value)[0]
# Start indices for each sample
index = tf.range(0, batch_size) * maxSeqLength + (seqlen - 1)
# Indexing
last= tf.gather(tf.reshape(value, [-1, lstmUnits]), index)
#last = tf.gather(value, int(value.get_shape()[0]) - 1)#size of the timestep -1 =last timestep
last = tf.cast(last, tf.float64)
bias = tf.cast(bias, tf.float64)
weight = tf.cast(weight, tf.float64)
#prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)# try logistic regression
#logistic regression
prediction = tf.nn.sigmoid(tf.matmul(last, weight) + bias)

#test
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float64))
#change
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=(tf.matmul(last, weight) + bias), labels=labels))
#optimizer = tf.train.AdamOptimizer().minimize(loss)
#change
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

#change add a variable seqlen
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels, nextseqlen = getTrainBatch(train_data,ids,batchSize,maxSeqLength);
   print(nextBatch)
   print(nextBatchLabels)
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels,seqlen: nextseqlen})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels,seqlen: nextseqlen})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 1000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
writer.close()


# for j in range(10):
#     for i in range(len(train_data)):
#         # Next Batch of reviews
#         nextBatchLabels = []
#         nextBatch = ids[i:i+1]
#         if train_data[i][0] == 'realDonaldTrump':
#             nextBatchLabels.append([1, 0])
#         else:
#             nextBatchLabels.append([0, 1])
#
#         print(train_data[i])
#
#         sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#
#         # Write summary to Tensorboard
#         if (i % 50 == 0):
#             summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#             writer.add_summary(summary, i)
#
#         # Save the network every 10,000 training iterations
#         if (i % 1000 == 0 and i != 0):
#             save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
#             print("saved to %s" % save_path)
#     j = j+1
#
# writer.close()


test_data = data_process('test.csv')
test_ids = get_idmatrix(test_data,word_list)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

for i in range(len(test_data)):
    nextBatch = test_ids[i:i + batchSize]
    # nextBatchLabels = np.array
    print(sess.run(prediction, {input_data: nextBatch}))
    i = i+batchSize
