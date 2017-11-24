#coding:utf-8

from __future__ import unicode_literals
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

def variable_summaries(var):
    '''
    计算执行过程中的参数变化
    Args:
    Returns:
    '''
    with tf.name_scope('summaries'):

        mean = tf.reduce_mean(var)            #计算参数的平均值
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))   #计算参数的标准差

        #记录参数值
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
 
        #用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size,
             sequence_length,vocab_size,embed_size,decay_steps, decay_rate, is_decay, is_dropout, is_l2):
        '''
        Args:
          filter_size:卷积核大小每次能覆盖的单词数(list)
          num_filters:每个卷积核的数量(int)
          num_classes:最后一层网络节点数(int)
          learning_rate:每次梯度下降的学习率(float)
          batch_size:每次训练的样本数(int)
          sequence_length:统一句子输入的长度(int)
          vocab_size:word2vec词汇量的大小，最终确定embedding层的大小
          embed_size:每个词的词向量长度
        Returns:
          None 
        '''
        print('初始化训练超参数.....')
        self.num_classes = num_classes            #分类个数

        self.filter_sizes = filter_sizes            #卷积参数
        self.num_filters = num_filters

        self.learning_rate = learning_rate        #训练参数
        self.batch_size = batch_size

        self.sequence_length = sequence_length    #词向量参数
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.decay_steps = decay_steps           #动态更新学习率
        self.decay_rate = decay_rate
        self.is_decay = is_decay

        self.is_l2 = is_l2

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  #dropout参数
        self.is_dropout = is_dropout

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
 
        print('定义占位符，输入输出变量.....')
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None,], name="input_y")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        self.initializer = tf.random_normal_initializer(stddev=0.1)

        print("定义网络结构，操作......")
        self.instantiate_weights()                #初始化权重
        self.logits = self.inference()            #构建图表网络信息，返回预测结果
        
        self.loss_val = self.loss()               #构建图表损失操作
        self.train_op = self.train()              #通过梯度下降来最小化损失函数的操作
     
        with tf.name_scope("Prediction"): 
            self.predictions = tf.argmax(self.logits, 1, name="predictions")  #返回logits列表中每一行的最大值
   
        with tf.name_scope("Accurary"):
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    def instantiate_weights(self):
        '''
        初始化网络参数
        Args:
          Embedding:[self.vocab_size, self.embed_size]
          W_projection:[self.num_filter_total, self.num_classes]
          b_projection:[self.num_classes]
        '''
        with tf.name_scope("Variables"):
            with tf.name_scope("Embedding"):
                self.Embedding = tf.get_variable("Embedding", 
                                   shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
            with tf.name_scope("W_projection"):     #计算输出层的参数
                self.W_projection = tf.get_variable("W_projection",
                                    shape=[self.num_filters_total, self.num_classes], initializer=self.initializer)
                variable_summaries(self.W_projection)

            with tf.name_scope("b_projection"):
                self.b_projection = tf.get_variable("b_projection", 
                                     shape=[self.num_classes])
                variable_summaries(self.b_projection)
    
    def inference(self):
        '''
        构建网络结构
        Args:
          Conv.Input:[filter_height, filter_width, in_channels, out_channels]
          Conv.Returns:[batch_size,sequence_length-filter_size+1,1,num_filters]
          input_data:NHWC:[batch, height, width, channels]
      
          pool.Input:[batch, height, width, channels]
        Returns:
          网络结构每次训练返回的结果:[batch_size, self.num_classes]
        '''
        with tf.name_scope("Layer_Embedding"):
            #[None, sentence_length, embed_size]
            self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
            self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1,
                                  name="embedding_word")  #[None, sentence_length, embed_size, 1]
 
        pooled_outputs = []
        with tf.name_scope("Conv2d"):
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("convolution-%s" %filter_size):
                    filter = tf.get_variable("filter-%s"%filter_size, 
                             [filter_size,self.embed_size,1,self.num_filters], initializer=self.initializer)
                    #[batch_size, self.sequence_size-filter_size, 1, 1]
                    conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter,
                                       strides=[1,1,1,1], padding="VALID", name="conv")
                with tf.name_scope("relu-%s"%filter_size):
                    b = tf.get_variable("b-%s"%filter_size, [self.num_filters])
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                
                with tf.name_scope("pool-%s"%filter_size):
                    pooled = tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1],
                                    strides=[1,1,1,1], padding="VALID", name="pool")

                pooled_outputs.append(pooled)

        with tf.name_scope("Pool_Flat"):
            self.h_pool = tf.concat(pooled_outputs,3) #[batch_size, 1, 1, num_filters_total]
            self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filters_total])

        if self.is_dropout:
            print('需要dropout操作')
            with tf.name_scope("DropOut"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob) 
            
        with tf.name_scope("Output"):
            #tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.matmul(self.h_pool_flat, self.W_projection) + self.b_projection  #[None, self.num_classes]
        return logits

    def loss(self, l2_lambda=0.0001):
        '''
        根据每次训练的预测结果和标准结果比较，计算误差
        loss = loss + l2_lambda*1/2*||variables||2
        Args:
            l2_lambda:超参数，l2正则，保证l2_loss和train_loss在同一量级
        Returns:
          每次训练的损失值loss
        '''
        with tf.name_scope("Loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
#            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            
            if self.is_l2:
                print("需要对loss进行l2正则化")
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
                loss = loss + l2_losses

            variable_summaries(loss)
   
        return loss

    def train(self):
        '''
        通过梯度下降最小化损失loss的操作
        Args:
        Returns:
          返回包含了训练操作(train_op)输出结果的tensor
        '''
        if self.is_decay:
            print("需要对学习率进行指数衰减")
            with tf.name_scope("LearningRate"):
                #学习率指数衰减 learning_rate=learning_rate*decay_rate^(global_step/decay_steps)
                learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                           self.decay_steps, self.decay_rate, staircase=True)

        with tf.name_scope("Train"):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
#        train_op = tf.contrib.layers.optimize_loss()
    
        return train_op
  
def main():
    filtersize = [1,2,3] 
    num_filters = 128
    num_classes = 3
    learning_rate = 0.01
    batch_size = 6
    sequence_length = 30
    vocab_size = 1000
    embed_size = 100

    textCNN = TextCNN(filtersize, num_filters, num_classes, learning_rate,
                     batch_size, sequence_length, vocab_size, embed_size)       
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size, sequence_length))
            input_y = np.array([1,0,1,1,1,2])
 
            feed_dict = {textCNN.input_x:input_x, textCNN.input_y:input_y}
            loss,acc,predict,W_projection_value,_ = sess.run([textCNN.loss_val,textCNN.accuracy,
                      textCNN.predictions, textCNN.W_projection, textCNN.train_op],feed_dict=feed_dict)

            print("loss:",loss,"acc:",acc,"lable:",input_y, "prediction:",predict)
if __name__ == "__main__":
    main()
