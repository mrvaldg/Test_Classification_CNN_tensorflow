import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import io
from tensorflow.contrib import learn


negative_examples=list()
positive_examples=list()
path ='train/pos'
for file in os.listdir(path):
    with io.open(os.path.join(path, file), 'r',encoding="utf8") as infile:
        txt = infile.read()
        positive_examples.append(txt)
path ='train/neg'
for file in os.listdir(path):
    with io.open(os.path.join(path, file), 'r',encoding="utf8") as infile:
        txt = infile.read()
        negative_examples.append(txt)
x_text=positive_examples+negative_examples
positive_labels=[]
negative_labels=[]
for i in positive_examples:
    positive_labels.append([0,1])
for j in negative_examples:
    negative_labels.append([1,0])
y = np.concatenate([positive_labels, negative_labels], 0)
vocab_processor = learn.preprocessing.VocabularyProcessor(max([len(x.split(" ")) for x in x_text]))
x = np.array(list(vocab_processor.fit_transform(x_text)))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

class Text_Classification_CNN(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, [None, x_train.shape[1]])
        self.input_y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), 128], -1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        pooled_outputs = []
        for i, filter_size in enumerate([3,4,5]):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, 128, 1, 128]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[128]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID")
                h = tf.nn.tanh(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(h,ksize=[1, x_train.shape[1] - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID')
                pooled_outputs.append(pooled)
        num_filters_total = 128 * 3
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, y_train.shape[1]],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[y_train.shape[1]]))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b)
            self.predictions = tf.argmax(self.scores, 1)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + 0.0 * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

print("Number of filters:",128,"\n","L2 Regularization:",0.0,"\n","Embedding dimension:",128,"\n","Number of epoch:",10,"\n","Batch size:",100,"\n","Filter sizes:","3,4,5","\n","Vocabulary Size:",len(vocab_processor.vocabulary_),"\n","Train/Test Sizes:",(len(y_train), len(y_test)))
with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = Text_Classification_CNN()
        global_step = tf.Variable(0,trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.global_variables_initializer())
        def batches_( file1,file2,batch_size, num_epochs):
            data = np.array(list(zip(file1,file2)))
            data_size = len(list(zip(file1,file2)))
            num_batches_per_epoch = int((len(list(zip(file1, file2))) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                shuffled_data = data[np.random.permutation(np.arange(data_size))]
                for batch_num in range(num_batches_per_epoch):
                    yield shuffled_data[batch_num * batch_size:min((batch_num + 1) * batch_size, data_size)]
        batches_text = batches_(x_test, y_test,100,2)
        batches_train=batches_(x_train,y_train,100,2)
        print("For Training Set")
        for batch in batches_train:
            x_batch, y_batch = zip(*batch)
            _, epoch, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],feed_dict = {cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.dropout_keep_prob: 0.5})
            print("epoch:",epoch,"loss:",loss,"accuracy:",accuracy)
            current_step = tf.train.global_step(sess, global_step)
        print("For Test Set")
        for batch2 in batches_text:
            x_batch_test, y_batch_test = zip(*batch2)
            epoch, loss, accuracy = sess.run([global_step,  cnn.loss, cnn.accuracy],feed_dict={cnn.input_x: x_batch_test,cnn.input_y: y_batch_test,cnn.dropout_keep_prob: 1.0})
            print("epoch",epoch,"loss:",loss,"accuracy:",accuracy)

