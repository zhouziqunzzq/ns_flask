# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
import random

from HyperParameter import HyperParameter
from data_helpers import *
from model import Seq2SeqModel


class Gen():

    def __init__(self):

        self.hp = HyperParameter()

        sources = load_and_cut_data(self.hp.sources_txt)
        targets = load_and_cut_data(self.hp.targets_txt)

        self.sources_data, self.targets_data, self.word_to_id, self.id_to_word = create_dic_and_map(sources, targets)

        self.model = Seq2SeqModel(
            rnn_size=self.hp.rnn_size,
            num_layers=self.hp.num_layers,
            embedding_size=self.hp.embedding_size,
            word_to_id=self.word_to_id,
            mode='predict',
            learning_rate=self.hp.learning_rate,
            use_attention=True,
            beam_search=True,
            beam_size=self.hp.beam_size,
            encoder_state_merge_method=self.hp.encoder_state_merge_method,
            max_gradient_norm=5.0
        )

    def restore_model(self, model_path):
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(model_path)
        self.model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict_ids_to_seq(self, predict_ids, id2word, beam_size, return_best=False):
        """
        将beam_search返回的结果转化为字符串
        :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
        :param id2word: vocab字典
        :return:
        """
        if return_best:
            index = 0
        else:
            index = random.randint(0, self.hp.beam_size - 1)
        sentence_list = []
        for single_predict in predict_ids:
            for i in range(beam_size):
                # print("Beam search result {}：".format(i + 1))
                predict_list = np.ndarray.tolist(single_predict[:, i])
                predict_seq = [id2word[idx] for idx in predict_list]
                sentence_list.append(predict_seq)
        return sentence_list[index]

    def user_input(self, input_text, sample_size):

        self.text = input_text
        self.sample_size = sample_size

    def generator(self):
        for i in range(self.sample_size):
            batch = sentence2enco(self.text, self.word_to_id)
            predict_ids = self.model.infer(self.sess, batch)
            self.text = self.predict_ids_to_seq(predict_ids, self.id_to_word, self.hp.beam_size)
            self.text = ''.join(''.join(reversed(self.text)).split('<EOS>'))
            sentence = self.text

            yield sentence

    def generate_next_split(self):
        batch = sentence2enco(self.text, self.word_to_id)
        predict_ids = self.model.infer(self.sess, batch)
        tmp = self.predict_ids_to_seq(predict_ids, self.id_to_word,
                                      self.hp.beam_size, return_best=False)
        rst = []
        for w in reversed(tmp):
            if w != '<EOS>':
                rst.append(w)

        return rst


def get_sentences(gen, input_text, sample_size):
    """
    返回指定行数的歌词（不保证押韵）
    :param gen: model
    :param input_text: 由主题词生成的一句文本
    :param sample_size: 要生成的歌词行数
    :return: 生成的歌词
    """
    lyrics = []

    gen.user_input(input_text, sample_size)
    sentences = gen.generator()
    for sen in sentences:
        lyrics.append(sen)

    return lyrics


def get_next_sentence_split(gen, input_text):
    """
    返回一行分词的歌词（不保证押韵）
    :param gen: model
    :param input_text: 由主题词生成的一句文本
    :return: 生成的歌词
    """
    gen.user_input(input_text, 1)
    return gen.generate_next_split()


if __name__ == '__main__':
    gen = Gen()
    gen.restore_model('model/')
    input_text = '腿搁在办公桌上'
    lyrics = get_sentences(gen, input_text, sample_size=10)

    for ly in lyrics:
        print(ly)
    #print(get_next_sentence_split(gen, input_text))
