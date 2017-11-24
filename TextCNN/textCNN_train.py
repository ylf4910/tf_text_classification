#coding:utf-8

from __future__ import unicode_literals
from __future__ import print_function

import os
import sys
sys.path.append("../util/")

import tensorflow as tf
import numpy as np
import pickle

from textCNN_model import TextCNN
from data_util import load_data_multilabel_new,create_vocabulary,create_vocabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
from gensim.models import KeyedVectors

#from p7_TextCNN_train import assign_pretrained_word_embedding

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 4, "类别数")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "学习率")
tf.app.flags.DEFINE_integer("num_epochs", 4, "训练迭代次数")
tf.app.flags.DEFINE_integer("batch_size", 50, "同时处理样本数")
tf.app.flags.DEFINE_integer("sequence_length", 50, "统一句子长度")

tf.app.flags.DEFINE_integer("embed_size", 100, "词向量长度")
tf.app.flags.DEFINE_string("training_data_path", "../data/train.txt", "训练数据路径")
tf.app.flags.DEFINE_string("word2vec_model_path", "../data/autohome.bin", "词向量")
tf.app.flags.DEFINE_boolean("use_embedding", False, "是否使用预训练的词向量模型")

tf.app.flags.DEFINE_string("ckpt_dir", "checkpoint/", "模型保存的目录地址")
tf.app.flags.DEFINE_string("tensorboard_dir", "logs/", "tensorboard事件存储目录地址")

tf.app.flags.DEFINE_integer("decay_steps", 5000, "learning_rate更新周期，每隔多少step更新一次learning rate的值")
tf.app.flags.DEFINE_float("decay_rate", 0.65, "衰减参数(对应α^t中的α)")
tf.app.flags.DEFINE_boolean("is_decay", True, "learning_rate是否动态更新")

tf.app.flags.DEFINE_boolean("is_dropout", True, "是否进行dropout")
tf.app.flags.DEFINE_boolean("is_l2", True, "是否对loss进行正则化")

tf.app.flags.DEFINE_integer("validate_every", 1, "每隔几轮即做一次验证")

tf.app.flags.DEFINE_integer("num_filters", 128, "每个尺寸的卷积核个数")
filter_size = [2,3,4]       #卷积核尺寸


def main(_):
    '''
    主函数：数据预处理，迭代数据训练模型
    '''
    print("数据预处理阶段：......")
    trainX,trainY,testX,testY = None, None, None, None
 
    #加载词向量转换
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(
                word2vec_model_path=FLAGS.word2vec_model_path, name_scope="cnn")

    vocab_size = len(vocabulary_word2index)
    #标签索引
    vocabulary_word2index_label, vocabulary_index2word_label = create_vocabulary_label(
               vocabulary_label=FLAGS.training_data_path, name_scope="cnn")

    #将文本转换为向量形式，分训练，测试集
    train, test, _ = load_data_multilabel_new(vocabulary_word2index, 
                   vocabulary_word2index_label,training_data_path=FLAGS.training_data_path)

    trainX, trainY = train
    testX, testY = test

    #用0填充短句
    trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0)
    testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0)
    print("数据预处理部分完成.....")

    print("创建session 对话.......")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        textCNN = TextCNN(filter_size,FLAGS.num_filters,FLAGS.num_classes,FLAGS.learning_rate,FLAGS.batch_size,
  FLAGS.sequence_length,vocab_size,FLAGS.embed_size,FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.is_decay, FLAGS.is_dropout,FLAGS.is_l2)

        #存储变量
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)

        #初始化模型保存
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):         #判断模型是否存在
            print("从模型中恢复变量")
#            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))   #自动获取最近一次的模型变量
        else:
            print("初始化变量")
            sess.run(tf.global_variables_initializer())     #初始化所有变量
            if FLAGS.use_embedding:        #加载预训练词向量
                assign_pretrained_word_embedding(sess, vocabulary_index2word,
                           vocab_size, textCNN, word2vec_model_path=FLAGS.word2vec_model_path)

        curr_epoch = sess.run(textCNN.epoch_step)
   
        #划分训练数据
        num_train_data = len(trainX)
        batch_size = FLAGS.batch_size

        index = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss,acc,counter = 0.0, 0.0, 0.0
            for start, end in zip(range(0, num_train_data, batch_size), range(batch_size, num_train_data, batch_size)):
                feed_dict = {textCNN.input_x:trainX[start:end], textCNN.input_y:trainY[start:end], 
                                   textCNN.dropout_keep_prob:0.9}
                curr_loss, curr_acc, logits, _ = sess.run([textCNN.loss_val, textCNN.accuracy, 
                                                textCNN.logits, textCNN.train_op], feed_dict)
     
                index += 1
                loss, counter, acc = loss+curr_loss, counter+1, acc+curr_acc
   
                if counter % 100 == 0:
                    rs = sess.run(merged,feed_dict)       #执行参数记录
                    writer.add_summary(rs, index)
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f\tGlobal Step %d"
                          %(epoch,counter,loss/float(counter),acc/float(counter),sess.run(textCNN.global_step)))
 #                   print("Train Logits{}".format(logits))
    
            #迭代次数增加
            epoch_increment = tf.assign(textCNN.epoch_step, tf.add(textCNN.epoch_step, tf.constant(1)))
            sess.run(epoch_increment)

            #验证
            print("迭代次数:{}".format(epoch))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess,textCNN,testX,testY,batch_size)
                print("迭代次数:{}\t验证损失值:{}\t准确率:{}".format(epoch, eval_loss, eval_acc))

                #保存模型
        #        save_path = FLAGS.ckpt_dir+"model.ckpt"
        #        saver.save(sess, save_path, global_step=epoch)

        print("验证集上进行损失，准确率计算.....")
        test_loss, test_acc = do_eval(sess, textCNN, testX, testY, batch_size)
        print("测试集中损失值:{}\t准确率:{}".format(test_loss, test_acc))

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path=None):
    """
    加载预训练的词向量，用于Embedding层
    Args:
      sess:会话
      vocabulary_index2word:词向量 索引-词语对
      vocab_size:词汇量
      textCNN:网络模型参数
      word2vec_model_path:词向量模型路径
    Returns:
    """
    print("使用预训练的词向量模型，加载模型{}".format(word2vec_model_path))
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True,
                                               encoding="utf-8", unicode_errors="ignore")
    word2vec_dict = {}
    for index,word in enumerate(word2vec_model.vocab.keys()):        #得到词向量的所有词汇
        if word2vec_model[word].shape[0] != 100:
            print(word2vec_model[word].shape)
            continue
        word2vec_dict[word] = word2vec_model[word]

    #初始化所有词向量，[vocab_size, embed_size]
    word_embedding_2dlist = [[]] * vocab_size    #创建一个空的词向量list:[[],[],[]....]
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)    #初始化词向量的第一个词语PAD为0
 
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)       #绑定一个随机变量
 
    count_exist = 0
    count_not_exist = 0

    for index in range(1, vocab_size):
        word = vocabulary_index2word[index]

        embedding = None
        try:
            embedding = word2vec_dict[word]
        except Exception as e:
            embedding = None

        if embedding is not None:    #当前词语存在一个词向量
            word_embedding_2dlist[index] = embedding
            count_exist = count_exist + 1            #设置当前词语的词向量
        else:
            word_embedding_2dlist[index] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1

    word_embedding_array = np.array(word_embedding_2dlist)    #将list转换为array
    word_embedding_tensor = tf.constant(word_embedding_array, dtype=tf.float32)

    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding_tensor)  #词向量添加到模型中的Embedding层

    sess.run(t_assign_embedding)
    print("存在词向量的词语数:{}\t不存在词向量的词语数:{}".format(count_exist, count_not_exist))
    print("加载预训练模型完成")
  
def do_eval(sess,textCNN,evalX,evalY,batch_size):
    '''
    在验证集上进行测试，计算损失及准确率
    Args:
      sess:会话
      textCNN:模型
      evalX:验证集数据
      evalY:验证集标签
      batch_size:批处理大小
    Returns:
      loss:验证集上的损失
      accuracy:验证集上的准确率
    '''
    num_examples = len(evalX)
    eval_loss, eval_counter, eval_acc = 0.0, 0.0, 0.0

    for start, end in zip(range(0,num_examples,batch_size), range(batch_size, num_examples, batch_size)):
        feed_dict = {textCNN.input_x:evalX[start:end], textCNN.input_y:evalY[start:end],
                           textCNN.dropout_keep_prob:1}
        curr_eval_loss, logits, curr_eval_acc = sess.run([textCNN.loss_val, textCNN.logits,textCNN.accuracy],feed_dict)

        eval_loss,eval_acc,eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1

    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)

if __name__ == "__main__":
    tf.app.run()
