#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : runserver.py.py
# @Author: harry
# @Date  : 18-8-19 下午5:56
# @Desc  : Run flask server

from gevent import monkey

monkey.patch_all()
import os
import _thread
import time
from flask import Flask, request, jsonify
from werkzeug.contrib.cache import SimpleCache
from gevent import pywsgi
import configparser
from predict import *

app = Flask(__name__)

model = None


@app.route('/generate/freestyle', methods=['POST'])
def generate_freestyle():
    if request.method == 'POST':
        # POST params
        text = str(request.form['text'])
        num_sentence = int(request.form['num_sentence'])

        # now return the first sentence along with generated sentences
        global model
        sentences = [text] + get_sentences(model, text, sample_size=num_sentence - 1)

        return jsonify(sentences)
    return 'POST method is required'


@app.route('/generate/next_sentence', methods=['POST'])
def generate_next_sentence():
    if request.method == 'POST':
        # POST params
        text = str(request.form['text'])

        # now return the first sentence along with generated sentences
        global model

        return jsonify(get_next_sentence_split(model, text))
    return 'POST method is required'


if __name__ == "__main__":
    # load model
    print("Loading model...")
    model = Gen()
    model.restore_model('./model')

    # load config from web.ini
    cp = configparser.ConfigParser()
    cp.read('web.ini')
    ip = str(cp.get('web', 'ip'))
    port = int(cp.get('web', 'port'))

    # start flask server
    print("Starting web server at {}:{}".format(ip, port))
    server = pywsgi.WSGIServer((ip, port), app)
    server.serve_forever()
