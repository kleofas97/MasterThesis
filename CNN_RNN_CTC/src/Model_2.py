import os
import sys

import numpy as np
import tensorflow as tf

"""Second version - Gated. Fully implemented on Colab due to calculations size - only trials here"""


# Disable eager mode
tf.compat.v1.disable_eager_execution()


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    "minimalistic TF model for HTR"

    # model constants
    imgSize = (512, 48)
    maxTextLen = 128

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
        "init model: add CNN, RNN and CTC and initialize TF"
        self.dump = dump
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def setupCNN(self):
        "create CNN layers and return output of these layers"

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)


        kernel = tf.Variable(tf.random.truncated_normal([3,3,1,8],stddev=0.1))
        conv = tf.nn.conv2d(input=cnnIn4d, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
        tanh = tf.nn.tanh(conv)
        kernel = tf.Variable(tf.random.truncated_normal([2,4,8,16],stddev=0.1))
        conv = tf.nn.conv2d(input=tanh, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
        tanh = tf.nn.tanh(conv)
        kernel = tf.Variable(tf.random.truncated_normal([3, 3,16, 16], stddev=0.1))
        conv_gate = tf.nn.conv2d(input=tanh, filters=kernel, padding='SAME', strides=(1, 1, 1,1))
        sigmoid = tf.nn.sigmoid(conv_gate)
        output = tf.multiply(sigmoid, tanh)
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 16, 32], stddev=0.1))
        conv = tf.nn.conv2d(input=output, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
        tanh = tf.nn.tanh(conv)
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 32,32], stddev=0.1))
        conv_gate = tf.nn.conv2d(input=tanh, filters=kernel, padding='SAME', strides=(1, 1, 1,1))
        sigmoid = tf.nn.sigmoid(conv_gate)
        output = tf.multiply(sigmoid, tanh)
        kernel = tf.Variable(tf.random.truncated_normal([2, 4, 32, 64], stddev=0.1))
        conv = tf.nn.conv2d(input=output, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
        tanh = tf.nn.tanh(conv)
        kernel = tf.Variable(tf.random.truncated_normal([3, 3,64, 64], stddev=0.1))
        conv_gate = tf.nn.conv2d(input=tanh, filters=kernel, padding='SAME', strides=(1, 1, 1,1))
        sigmoid = tf.nn.sigmoid(conv_gate)
        output = tf.multiply(sigmoid, tanh)
        kernel = tf.Variable(tf.random.truncated_normal([3, 3, 64, 128], stddev=0.1))
        conv = tf.nn.conv2d(input=output, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
        tanh = tf.nn.tanh(conv)

        pool = tf.nn.max_pool2d(input=tanh, ksize=(1, 1, Model.imgSize[1], 1),
                                strides=(1, 1, 1, 1), padding='VALID')

        self.cnnOut4d = pool

    def setupRNN(self):
        "create RNN layers and return output of these layers"


        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=numHidden, state_is_tuple=True) for _
                 in
                 range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked,
                                                                  inputs=rnnIn3d,
                                                                  dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)



        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.random.truncated_normal([1, 1, numHidden*2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setupCTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(a=self.rnnOut3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                       tf.compat.v1.placeholder(tf.int32, [None]),
                                       tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC,
                                                                         sequence_length=self.seqLen,
                                                                         ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[Model.maxTextLen, None, len(self.charList) + 1])
        self.lossPerElement = tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput,
                                                       sequence_length=self.seqLen, ctc_merge_repeated=True)

        # best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                                                         beam_width=50)
        # word beam search decoding (see https://github.com/githubharald/CTCWordBeamSearch)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)
            wordChars = open('../model2/wordCharList.txt').read().splitlines()[0]
            corpus = open('../data/corpus.txt').read()

            # decode using the "Words" mode of word beam search
            from word_beam_search import WordBeamSearch
            self.decoder = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                          wordChars.encode('utf8'))

            # the input to the decoder must have softmax already applied
            self.wbsInput = tf.nn.softmax(self.ctcIn3dTBC, axis=2)

    def setupTF(self):
        "initialize TF"
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        modelDir = r'F:/Studia/pythonProject/TestHDR/SimpleHTR/model2/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):
        "extract texts from output of CTC decoder"

        # word beam search: already contains label strings
        if self.decoderType == DecoderType.WordBeamSearch:
            labelStrs = ctcOutput

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # contains string of labels for each batch element
            labelStrs = [[] for _ in range(batchSize)]

            # go over all indices and save mapping: batch -> values
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                labelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in labelStrs]

    def trainBatch(self, batch):
        "feed a batch into the NN to train it"
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs: batch.imgs, self.gtTexts: sparse,
                    self.seqLen: [Model.maxTextLen] * numBatchElements, self.is_train: True}
        _, lossVal = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal

    def dumpNNOutput(self, rnnOutput):
        "dump the output of the NN to CSV file(s)"
        dumpDir = '../dump/'
        if not os.path.isdir(dumpDir):
            os.mkdir(dumpDir)

        # iterate over all batch elements and create a CSV file for each one
        maxT, maxB, maxC = rnnOutput.shape
        for b in range(maxB):
            csv = ''
            for t in range(maxT):
                for c in range(maxC):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            fn = dumpDir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        "feed a batch into the NN to recognize the texts"

        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)

        # put tensors to be evaluated into list
        evalList = []

        if self.decoderType == DecoderType.WordBeamSearch:
            evalList.append(self.wbsInput)
        else:
            evalList.append(self.decoder)

        if self.dump or calcProbability:
            evalList.append(self.ctcIn3dTBC)

        # dict containing all tensor fed into the model
        feedDict = {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements,
                    self.is_train: False}

        # evaluate model
        evalRes = self.sess.run(evalList, feedDict)

        # TF decoders: decoding already done in TF graph
        if self.decoderType != DecoderType.WordBeamSearch:
            decoded = evalRes[0]
        # word beam search decoder: decoding is done in C++ function compute()
        else:
            decoded = self.decoder.compute(evalRes[0])

        # map labels (numbers) to character string
        texts = self.decoderOutputToText(decoded, numBatchElements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput: ctcInput, self.gtTexts: sparse,
                        self.seqLen: [Model.maxTextLen] * numBatchElements, self.is_train: False}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dumpNNOutput(evalRes[1])

        return texts, probs

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, '../model2/snapshot', global_step=self.snapID)
