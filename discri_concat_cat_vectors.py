
# coding: utf-8


import numpy as np

from utility import load_data
import random
import pandas as pd
from time import time
from sklearn import metrics
from sample_encoding import *
from cat2vec import Options, Cat2Vec
import tensorflow as tf

def read_file(path, infinite=True):
    while True:
        fi = open(path,'r')
        for line in fi:
            yield map(int,line.replace('\n', '').split(' '))
        if infinite == False:
            break

def get_substitute_cate(sample, target_index, opts):
    field_i = opts.fields_index_inverse.get(sample[target_index])
    if field_i is None:
        field_i = np.random.choice(opts.fields_index.keys(),1)[0]
    field_cates = opts.fields_index[field_i]
    rst = np.random.choice(field_cates,1)[0]
    if len(field_cates) == 1:
        rst = np.random.randint(opts.vocabulary_size)
    return rst

def generate_fake_sample(temp, opts):
        temp_sequence_length = len(temp)
        temp = temp[0:opts.sequence_length]
        if len(temp) < opts.sequence_length:
            gap = opts.sequence_length - len(temp)
            temp = np.array(temp + [0] * gap)
        else:
            temp_sequence_length = opts.sequence_length
        assert len(temp) == opts.sequence_length
        targets_to_avoid = set(temp)
        indices_to_avoid = set()
        substitute_index = np.random.randint(temp_sequence_length)
        substitute_target = get_substitute_cate(temp, substitute_index, opts)
        for _ in range(opts.substitute_num):
            while substitute_index in indices_to_avoid:
                substitute_index = np.random.randint(temp_sequence_length)
            indices_to_avoid.add(substitute_index)

            count = 0
            while substitute_target in targets_to_avoid:
                if count > 5:
                    break
                substitute_target = get_substitute_cate(temp, substitute_index, opts)
                count += 1
            targets_to_avoid.add(substitute_target)
            temp[substitute_index] = substitute_target
        return temp


def generate_discriminant_batch(opts, is_train=True, rate=0.5):
    data_index = 0
    if is_train:
        file_reader = read_file(opts.train_path)
    else:
        file_reader = read_file(opts.test_path)
    while True:
        batch = np.ndarray(shape=(opts.batch_size, opts.sequence_length))
        labels = np.ndarray(shape=(opts.batch_size, opts.num_classes))
        for i in xrange(opts.batch_size):
            target = np.zeros(opts.num_classes)
            if random.random() > rate:
                target[1] = 1.
                single_sample  = file_reader.next()
                temp = single_sample[1:opts.sequence_length]
                if len(temp) < opts.sequence_length:
                    gap = opts.sequence_length - len(temp)
                    temp = np.array(temp + [0] * gap)
                assert len(temp) == opts.sequence_length
                batch[i] = temp
                labels[i] = target
            else:
                target[0] = 1.
                single_sample  = file_reader.next()
                temp = single_sample[1:opts.sequence_length]
                batch[i] = generate_fake_sample(temp, opts)
                labels[i] = target
        yield batch, labels


# In[7]:

class DiscriminantCat2Vec(Cat2Vec):

    def __init__(self, options, session, cate2id, id2cate, pre_trained_emb=None, trainable=True, pre_trained_path=None):
        self.pre_trained_emb = None
        self.trainable = trainable
        if pre_trained_path is not None:
            self.load_pre_trained(pre_trained_path)
        Cat2Vec.__init__(self, options, session, cate2id, id2cate)
        # self.build_graph()

    def load_pre_trained(self, path):
        self.pre_trained_emb = np.array(pd.read_csv(path, sep=',',header=None),dtype=np.float32)
        print('pre-trained shape',self.pre_trained_emb.shape)

    def build_graph(self):
        """Build the model graph."""
        opts = self._options
        first_indices, second_indices =             get_batch_pair_indices(opts.batch_size, opts.sequence_length)
        # print(first_indices.shape)
        # the following is just for example, base class should not include this
        # with self._graph.as_default():
        self.train_inputs = tf.placeholder(tf.int32,
                                           shape=[opts.batch_size,
                                                  opts.sequence_length])
        self.train_labels = tf.placeholder(tf.int32, shape=[opts.batch_size,
                                                            opts.num_classes])
        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'):

            if self.pre_trained_emb is None:
                self.embeddings = tf.Variable(tf.random_normal([opts.vocabulary_size,
                                  opts.embedding_size],
                                 stddev=1.0 / np.sqrt(opts.embedding_size)
                                 ))
            else:
                if self.pre_trained_emb.shape == (opts.vocabulary_size, opts.embedding_size):
                    self.embeddings = tf.get_variable(name="embeddings",
                                                      shape=[opts.vocabulary_size, opts.embedding_size],
                                                      dtype=tf.float32,
                                                      initializer=tf.constant_initializer(self.pre_trained_emb),
                                                      trainable=self.trainable)
                    print('Inited by pre-trained embeddings')
                else:
                    print('pre_trained_emb shape', self.pre_trained_emb.shape )
                    print('vocabulary_size,embedding_size',(opts.vocabulary_size, opts.embedding_size))
                    raise Exception('Error', 'pre_trained_emb size mismatch')

            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            l2_loss += tf.nn.l2_loss(embed)
            encoded = tf.reshape(embed,[opts.batch_size,-1])
            # encoded = tf.concat(1,[encoded,tf.reshape(embed,[opts.batch_size,-1])])
            with tf.name_scope("output"):
                encoded_size = encoded.get_shape().as_list()[1]
                W, b = weight_bias([encoded_size, opts.num_classes], [
                                   opts.num_classes], bias_init=0.)
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.matmul(encoded, W) + b
                self.predictions = tf.argmax(scores, 1, name="predictions")

            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    scores, tf.to_float(self.train_labels))
                self.loss = tf.reduce_mean(losses) + opts.l2_reg_lambda * l2_loss

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(
                    self.predictions, tf.argmax(self.train_labels, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"), name="accuracy")

            optimizer =                 tf.train.GradientDescentOptimizer(opts.learning_rate)
#             self.loss = tf.clip_by_value(self.loss,-10,10)
#             optimizer = tf.train.AdamOptimizer()
            self.train_operator =                 optimizer.minimize(self.loss,
                                   gate_gradients=optimizer.GATE_NONE)
        tf.initialize_all_variables().run()
        print("Initialized")

    def eval_acc(self):
        opts = self._options
        if opts.test_num == 0:
            for line in open(opts.test_path,'r'):
                opts.test_num += 1
        batch_generator_test = generate_discriminant_batch(opts,is_train=False)
        batch_num = opts.test_num//opts.batch_size
        print('Total testing batch number', batch_num)
        average_loss = 0.
        acc = 0.
        start = time()
        for j in range(batch_num):
            batch_inputs, batch_labels = batch_generator_test.next()
            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}
            loss, accuracy = self._session.run([self.loss,
                                                self.accuracy],
                                                feed_dict=feed_dict)
            average_loss += loss
            acc += accuracy
        average_loss /= batch_num
        acc /= batch_num
        print("Test data, average loss: ", average_loss, ' accuracy: ', acc, 'total time for test', time()-start)

    def train(self, batch_generator, num_steps):
        opts = self._options
        average_loss = 0.
        acc = 0.
        start = time()
        for step in xrange(num_steps):
            batch_inputs, batch_labels = batch_generator.next()
            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}
            _, loss, accuracy = self._session.run([self.train_operator,
                                                   self.loss,
                                                   self.accuracy],
                                                  feed_dict=feed_dict)
            average_loss += loss
            acc += accuracy
            if step % 10000 == 0:
                t = time()-start
                if step > 0:
                    average_loss /= 10000
                    t  /= 10000
                    acc /= 10000
                print("Average loss at step ", step, ": ", average_loss,
                    ' accuracy: ', acc, 'time', t)
                average_loss = 0
            if step % 50000 == 0:
                print('Eval at step ', step)
                self.eval_clustering()
                self.eval_acc()
#         self.eval()


# In[3]:

opts = Options()
opts.sequence_length = 20
# opts.vocabulary_size = vocabulary_size
opts.norm_type = 'l2'
opts.gate_type = 'highway'
opts.batch_size = 32
opts.embedding_size = 32
opts.interaction_times = 2
opts.learning_rate = 0.1
opts.l2_reg_lambda = 0.01
opts.substitute_num = 3
opts.test_num = 0
# change here for different camp
# 1458 2261 2997 3386 3476 2259 2821 3358 3427 all
BASE_PATH = './data/make-ipinyou-data/all/'
opts.field_indices_path = BASE_PATH + 'field_indices.txt'
opts.train_path = BASE_PATH + 'train.yzx.txt'
opts.test_path = BASE_PATH + 'test.yzx.txt'
opts.featindex = BASE_PATH + 'featindex.txt'


# In[4]:

print('Loading data...')
opts.fields_index = {}
opts.fields_index_inverse = {}
f = open(opts.field_indices_path, 'r')
for line in f.readlines():
    field_name, indices = line.replace('\n','').split('\t')
    if indices != '':
        indices = np.array([int(i) for i in indices.split(',')])
    else:
        indices = [0]
    opts.fields_index[field_name] = indices
    for ind in indices:
        opts.fields_index_inverse[ind] = field_name


# In[5]:

vocabulary_size = 0
reverse_dictionary_raw = np.array(pd.read_csv(opts.featindex, sep='\t', header=None))
reverse_dictionary = {}
dictionary = {}
for item in reverse_dictionary_raw:
    reverse_dictionary[int(item[1])] = item[0]
    dictionary[item[0]] = int(item[1])
if item[1] > vocabulary_size:
    vocabulary_size = item[1]
vocabulary_size = len(dictionary.keys())
print('vocabulary_size: ',vocabulary_size)
id2cate = reverse_dictionary
cate2id = dictionary
opts.vocabulary_size = vocabulary_size
opts.id2cate = id2cate
opts.cate2id = cate2id


# In[10]:

batch_generator= generate_discriminant_batch(opts,is_train=True)

# pre_trained_path = './data/ipinyou/pre_trained_embs_72746_skip_cat_32.csv'
pre_trained_path = None
print('Building graph')
with tf.Graph().as_default(), tf.Session() as session:
    discr_cat2vec = DiscriminantCat2Vec(opts, session, id2cate, cate2id,
                                        pre_trained_emb=None,
                                        trainable=True,
                                        pre_trained_path=pre_trained_path)
    print('Training model')
    discr_cat2vec.train(batch_generator, 1000001)
    final_embeddings = discr_cat2vec.embeddings.eval()


# In[11]:

# final_embeddings = final_embedding
f = open(BASE_PATH + 'embedding_bag_of' + str(opts.embedding_size) + '_' + str(time()) + '.csv' ,'w')
for line in final_embeddings:
    f.write(','.join([str(l) for l in line])+'\n')
f.close()
print(np.array(final_embeddings).shape)

reverse_dictionary = {}
for i in range(len(id2cate)):
    reverse_dictionary[i] = id2cate[i]
dictionary = cate2id


# In[12]:

# get_ipython().magic(u'matplotlib inline')
figure_base_path = './figure/'
import matplotlib.patches as mpatches
def get_colors(cates):
    import six
    from matplotlib import colors
    colors_ = list(six.iteritems(colors.cnames))
    cates_set = list(set(cates))
    colors_mapping = {}
    for i in range(len(cates_set)):
        colors_mapping[cates_set[i]] = colors_[i][0]
    colors_mapping
    cates = [colors_mapping.get(c) for c in cates]
    return cates

def plot_with_labels(low_dim_embs, labels, cates, filename='all_500sample.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(30, 20))  #in inches
    cates_c = get_colors(cates)
    for color, cate in set(zip(cates_c,cates)):
        x, y = low_dim_embs[0,:]
        plt.scatter(x, y, c=color, label=cate, s=200, alpha=0.7, edgecolors='none')

    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y, c=cates_c[i], s=300, alpha=0.7, edgecolors='none')
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(0.8, 0.95),
                 textcoords='offset points',
                 ha='right',
                 va='top',
                    size=15)
    plt.legend()
    plt.savefig(figure_base_path + filename,format='pdf', )
    # plt.show()


# In[13]:

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    # reverse_dictionary[i].split('_')[-1][0:10]
    labels = ['' for i in xrange(plot_only)]
    cates = [opts.fields_index_inverse.get(i) for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, cates ,filename='all_500_dirc_bag_of' + BASE_PATH.split('/')[-2]+ '_' +str(opts.embedding_size) + '_' + str(time()) + '.pdf')

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")
