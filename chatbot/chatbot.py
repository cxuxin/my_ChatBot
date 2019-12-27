#!/usr/bin/env python
# encoding: utf-8
# @Software: PyCharm
# @Project: Chatbot_Seq2Seq
# @File: chatbot.py
# @Desc:代码主体
import argparse  # 命令行解析
import configparser  # 保存模型参数
import datetime
import math
import os  # 文件管理

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import tqdm  # 进度条

from chatbot.model import Model
from chatbot.textdata import TextData


class Chatbot:
    """
    启动测试还是训练模式的主类
    """

    class TestMode:
        """
        不同测试模式
        """
        ALL = 'all'
        INTERACTIVE = 'interactive'  # 用户可以写自己的问题
        DAEMON = 'daemon'  # 聊天机器人在后台运行，可以定期调用以预测某些事情

    def __init__(self):
        self.args = None  # 模型/数据集参数

        # 任务特定的对象
        self.textData = None  # 数据集
        self.model = None  # Seq2Seq模型

        # Tensorflow实用程序，方便保存/记录
        self.writer = None
        self.saver = None
        self.modelDir = ''  # 保存模型的位置
        self.globStep = 0  # 表示当前模型的迭代次数

        # TensorFlow主会话（跟踪进程）
        self.sess = None

        # 文件名和目录常量
        self.MODEL_DIR_BASE = 'save' + os.sep + 'model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.5'
        self.TEST_IN_NAME = 'data' + os.sep + 'test' + os.sep + 'samples.txt'
        self.TEST_OUT_SUFFIX = '_predictions.txt'
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

    @staticmethod
    def parseArgs(args):
        """
        从给定的命令行解析参数
        @param args: (list<str>) 要解析的参数列表。如果为None，则默认解析为sys.argv
        """

        parser = argparse.ArgumentParser()

        # 全局设置
        globalArgs = parser.add_argument_group('Global options')
        # '--test all/interactive/daemon'运行测试模式
        globalArgs.add_argument('--test',
                                nargs='?',
                                choices=[Chatbot.TestMode.ALL, Chatbot.TestMode.INTERACTIVE, Chatbot.TestMode.DAEMON],
                                const=Chatbot.TestMode.ALL, default=None,
                                help='if present, launch the program try to answer all sentences from data/test/ with'
                                     ' the defined model(s), in interactive mode, the user can wrote his own sentences,'
                                     ' use daemon mode to integrate the chatbot in another program')
        # '--createDataset'  将仅从语料库生成数据集（无需培训/测试）
        globalArgs.add_argument('--createDataset', action='store_true',
                                help='if present, the program will only generate the dataset from the corpus (no training/testing)')
        # '--playDataset'
        globalArgs.add_argument('--playDataset', type=int, nargs='?', const=10, default=None,
                                help='if set, the program  will randomly play some samples(can be use conjointly with createDataset if this is the only action you want to perform)')
        # '--reser' 如果要忽略模型目录中存在的先前模型，请使用此选项（警告：所有文件夹内容都会破坏该模型）
        globalArgs.add_argument('--reset', action='store_true',
                                help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
        # '--verbose' 测试时，将在计算输出的同时绘制输出
        globalArgs.add_argument('--verbose', action='store_true',
                                help='When testing, will plot the outputs at the same time they are computed')
        # '--debug' 在Tensorflow调试模式下运行。
        globalArgs.add_argument('--debug', action='store_true',
                                help='run DeepQA with Tensorflow debug mode. Read TF documentation for more details on this.')
        # '--keepAll' 保存所有已保存的模型（警告：请确保您有足够的可用磁盘空间或增加saveEvery
        globalArgs.add_argument('--keepAll', action='store_true',
                                help='If this option is set, all saved model will be kept (Warning: make sure you have enough free disk space or increase saveEvery)')  # TODO: Add an option to delimit the max size
        # '--modelTag' 标签以区分要存储/加载的模型
        globalArgs.add_argument('--modelTag', type=str, default=None,
                                help='tag to differentiate which model to store/load')
        # '--rootDir' 查找模型和数据的文件夹
        globalArgs.add_argument('--rootDir', type=str, default=None,
                                help='folder where to look for the models and data')
        # '--watsonMode' 反转问题并在训练时回答（网络尝试猜测问题）
        globalArgs.add_argument('--watsonMode', action='store_true',
                                help='Inverse the questions and answer when training (the network try to guess the question)')
        # '--autoEncode' 随机选择问题或答案并将其用作输入和输出
        globalArgs.add_argument('--autoEncode', action='store_true',
                                help='Randomly pick the question or the answer and use it both as input and output')
        # '--device' \'gpu \'或\'cpu \'（警告：确保您有足够的可用RAM），允许选择在哪个硬件上运行模型
        globalArgs.add_argument('--device', type=str, default=None,
                                help='\'gpu\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')
        # '--seed' 提取数据集的语料库
        globalArgs.add_argument('--seed', type=int, default=None, help='random seed for replication')

        # 数据集选项
        datasetArgs = parser.add_argument_group('Dataset options')
        # '--corpus' 提取数据集的语料库
        datasetArgs.add_argument('--corpus', choices=TextData.corpusChoices(), default=TextData.corpusChoices()[0],
                                 help='corpus on which extract the dataset.')
        # '--datasetTag' 向数据集添加标签（向其中加载词汇表和预先计算的样本的文件，而不是原始语料库）。对管理多个版本很有用。也用于定义用于轻量级格式的文件。如果样本不存在，则从语料库中计算样本。保存在\'data / samples / \'
        datasetArgs.add_argument('--datasetTag', type=str, default='',
                                 help='add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions. Also used to define the file used for the lightweight format.')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'
        datasetArgs.add_argument('--ratioDataset', type=float, default=1.0,
                                 help='ratio of dataset used to avoid using the whole dataset')  # Not implemented, useless ?
        # '--maxLength'  句子的最大长度（用于输入和输出），定义RNN的最大步数
        datasetArgs.add_argument('--maxLength', type=int, default=10,
                                 help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')
        # '--filterVocab' 删除很少使用的单词（默认情况下仅使用一次）。 0保留所有单词。
        datasetArgs.add_argument('--filterVocab', type=int, default=1,
                                 help='remove rarelly used words (by default words used only once). 0 to keep all words.')
        # '--skipLines' 仅通过使用偶数对话行作为问题（而奇数行作为答案）来生成训练样本。对于在特定人员上训练网络很有用。
        datasetArgs.add_argument('--skipLines', action='store_true',
                                 help='Generate training samples by only using even conversation lines as questions (and odd lines as answer). Useful to train the network on a particular person.')
        # '--vocabularySize' 限制词汇中的单词数（0表示无限制）
        datasetArgs.add_argument('--vocabularySize', type=int, default=40000,
                                 help='Limit the number of words in the vocabulary (0 for unlimited)')

        # 网络选项（警告：如果在此处进行修改，请同时在save / loadParams（）上进行更改)
        nnArgs = parser.add_argument_group('Network options', 'architecture related option')
        # '--hiddenSize' 每个RNN单元中隐藏单元的数量
        nnArgs.add_argument('--hiddenSize', type=int, default=512, help='number of hidden units in each RNN cell')
        # '--numLayers' rnn层数
        nnArgs.add_argument('--numLayers', type=int, default=2, help='number of rnn layers')
        # '--softmaxSamples' softmax损失函数中的样本数。值为0将停用采样的softmax
        nnArgs.add_argument('--softmaxSamples', type=int, default=0,
                            help='Number of samples in the sampled softmax loss function. A value of 0 deactivates sampled softmax')
        # '--initEmbeddings' 将使用预训练的word2vec向量初始化嵌入
        nnArgs.add_argument('--initEmbeddings', action='store_true',
                            help='if present, the program will initialize the embeddings with pre-trained word2vec vectors')
        # '--embeddingSize'  单词表示的嵌入大小
        nnArgs.add_argument('--embeddingSize', type=int, default=64, help='embedding size of the word representation')
        # '--embeddingSource' 嵌入文件以用于单词表示
        nnArgs.add_argument('--embeddingSource', type=str, default="GoogleNews-vectors-negative300.bin",
                            help='embedding file to use for the word representation')

        # Training options
        trainingArgs = parser.add_argument_group('Training options')
        # '--numEpochs' 运行的最大epoch数
        trainingArgs.add_argument('--numEpochs', type=int, default=30, help='maximum number of epochs to run')
        # '--saveEvery' 创建模型检查点之前的步骤数
        trainingArgs.add_argument('--saveEvery', type=int, default=2000,
                                  help='nb of mini-batch step before creating a model checkpoint')
        # '--batchSize' batch大小
        trainingArgs.add_argument('--batchSize', type=int, default=256, help='mini-batch size')
        # '--learningRate'  学习率
        trainingArgs.add_argument('--learningRate', type=float, default=0.002, help='Learning rate')
        # '--dropout' dropout (保持率)
        trainingArgs.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probabilities)')

        return parser.parse_args(args)

    def main(self, args=None):
        """
        启动训练或交互模式
        """
        print('Welcome to my ChatBot')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # 初始化
        self.args = self.parseArgs(args)

        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()  # 使用当前工作目录

        # tf.logging.set_verbosity(tf.logging.INFO) # DEBUG, INFO, WARN (default), ERROR, or FATAL

        self.loadModelParams()  # 现在更新self.modelDir和self.globStep，在加载模型时不使用（但需要在_getSummaryName之前调用）

        self.textData = TextData(self.args)

        if self.args.createDataset:
            print('Dataset created! Thanks for using my program')
            return  # No need to go further

        # 准备模型
        with tf.device(self.getDevice()):
            self.model = Model(self.args, self.textData)

        # Saver/summaries
        self.writer = tf.summary.FileWriter(self._getSummaryName())
        self.saver = tf.train.Saver(max_to_keep=200)

        # 运行session
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,  # 允许备份设备执行非GPU可用的操作（强制GPU时）
            log_device_placement=False)
        )

        if self.args.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        print('Initialize variables...')
        self.sess.run(tf.global_variables_initializer())

        # 最终重新加载模型（如果存在）。在测试模式下，未在此处加载模型（而是在predictTestset中）
        if self.args.test != Chatbot.TestMode.ALL:
            self.managePreviousModel(self.sess)

        # 使用预训练的word2vec向量初始化嵌入
        if self.args.initEmbeddings:
            self.loadEmbedding(self.sess)

        if self.args.test:
            if self.args.test == Chatbot.TestMode.INTERACTIVE:
                self.mainTestInteractive(self.sess)
            elif self.args.test == Chatbot.TestMode.ALL:
                print('Start predicting...')
                self.predictTestset(self.sess)
                print('All predictions done')
            elif self.args.test == Chatbot.TestMode.DAEMON:
                print('Daemon mode, running in background...')
            else:
                raise RuntimeError('Unknown test mode: {}'.format(self.args.test))  # Should never happen
        else:
            self.mainTrain(self.sess)

        if self.args.test != Chatbot.TestMode.DAEMON:
            self.sess.close()
            print("The End! Thanks for using my program")

    def mainTrain(self, sess):
        """
        训练循环
        @param sess: 正在运行的session
        """

        # Specific training dependent loading

        self.textData.makeLighter(self.args.ratioDataset)  # 限制训练样本数

        mergedSummaries = tf.summary.merge_all()  # 定义摘要运算符（警告：不会在张量板上显示）
        if self.globStep == 0:  # 无法从先前的运行还原
            self.writer.add_graph(sess.graph)  # 仅限第一次

        print('Start training (press Ctrl+C to save and exit)...')

        try:  # 如果用户在训练时退出，我们仍会尝试保存模型
            for e in range(self.args.numEpochs):

                print("\n")
                print("----- Epoch {}/{} ; (lr={}) -----".format(e + 1, self.args.numEpochs, self.args.learningRate))

                batches = self.textData.getBatches()

                tic = datetime.datetime.now()
                for nextBatch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feedDict = self.model.step(nextBatch)
                    assert len(ops) == 2  # training, loss
                    _, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)
                    self.writer.add_summary(summary, self.globStep)
                    self.globStep += 1

                    # Output training status
                    if self.globStep % 100 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.globStep, loss, perplexity))

                    # Checkpoint
                    if self.globStep % self.args.saveEvery == 0:
                        self._saveSession(sess)

                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(toc - tic))  # 警告：如果某个时间段超过24小时，将会溢出，并且输出效果不是很好
        except (KeyboardInterrupt, SystemExit):  # 如果用户在测试进度时按Ctrl + C
            print('Interruption detected, exiting the program...')

        self._saveSession(sess)  # Ultimate saving before complete exit

    def predictTestset(self, sess):
        """
        试着从samples.txt文件中预测句子。这些语句以相同的名称保存在modelDir中
        @param sess: The current running session
        """

        # 加载文件进行预测
        with open(os.path.join(self.args.rootDir, self.TEST_IN_NAME), 'r') as f:
            lines = f.readlines()

        modelList = self._getModelList()
        if not modelList:
            print('Warning: No model found in \'{}\'. Please train a model before trying to predict'.format(
                self.modelDir))
            return

        # 预测modelDir中存在的每个模型
        for modelName in sorted(modelList):  # TODO: Natural sorting
            print('Restoring previous model from {}'.format(modelName))
            self.saver.restore(sess, modelName)
            print('Testing...')

            saveName = modelName[:-len(
                self.MODEL_EXT)] + self.TEST_OUT_SUFFIX  # We remove the model extension and add the prediction suffix
            with open(saveName, 'w', encoding='utf-8') as f:
                nbIgnored = 0
                for line in tqdm(lines, desc='Sentences'):
                    question = line[:-1]  # 删除endl字符

                    answer = self.singlePredict(question)
                    if not answer:
                        nbIgnored += 1
                        continue

                    predString = '{x[0]}{0}\n{x[1]}{1}\n\n'.format(question,
                                                                   self.textData.sequence2str(answer, clean=True),
                                                                   x=self.SENTENCES_PREFIX)
                    if self.args.verbose:
                        tqdm.write(predString)
                    f.write(predString)
                print('Prediction finished, {}/{} sentences ignored (too long)'.format(nbIgnored, len(lines)))

    def mainTestInteractive(self, sess):
        """
        尝试预测用户将在控制台中输入的句子
        @param sess: 正在运行的session
        @return:
        """
        #  如果是verbose模式，则还显示来自训练集中的相似单词且带有相同单词的单词（也包括在mainTest中）

        print('Testing: Launch interactive mode:')
        print('')
        print('Welcome to the interactive mode, here you can ask to Deep Q&A the sentence you want. '
              ' Type \'exit\' or just press ENTER to quit the program.')

        while True:
            question = input(self.SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break

            questionSeq = []  # Will be contain the question as seen by the encoder
            answer = self.singlePredict(question, questionSeq)
            if not answer:
                print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
                continue

            print('{}{}'.format(self.SENTENCES_PREFIX[1], self.textData.sequence2str(answer, clean=True)))

            if self.args.verbose:
                print(self.textData.batchSeq2str(questionSeq, clean=True, reverse=True))
                print(self.textData.sequence2str(answer))

            print()

    def singlePredict(self, question, questionSeq=None):
        """
        预测句子
        @param question: （str）原始输入的句子
        @param questionSeq: （List <int>）输出参数。如果给定将包含输入批处理序列
        @return:<int> 对应答案的单词ID
        """
        # 创建输入batch
        batch = self.textData.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)

        # 运行模型
        ops, feedDict = self.model.step(batch)
        output = self.sess.run(ops[0], feedDict)  # TODO: Summarize the output too (histogram, ...)
        answer = self.textData.deco2sentence(output)

        return answer

    def daemonPredict(self, sentence):
        """
        返回给定句子的回答（与singlePredict（）相同，但附加清洁）
        @param sentence: （str）原始输入句子
        @return: 可读的句子
        """
        return self.textData.sequence2str(
            self.singlePredict(sentence),
            clean=True
        )

    def daemonClose(self):
        """
        完成时关闭程序的实用程序功能
        """
        print('Exiting the daemon mode...')
        self.sess.close()
        print('Daemon closed.')

    def loadEmbedding(self, sess):
        '''
        使用预训练的word2vec向量初始化嵌入
        将修改当前加载模型的嵌入权重
        使用GoogleNews预训练的值（路径为hardcoded）
        '''

        # 从模型中获取嵌入变量
        with tf.variable_scope("embedding_rnn_seq2seq/rnn/embedding_wrapper", reuse=True):
            em_in = tf.get_variable("embedding")
        with tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder", reuse=True):
            em_out = tf.get_variable("embedding")

        # Disable training for embeddings
        variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables.remove(em_in)
        variables.remove(em_out)

        # If restoring a model, we can leave here
        if self.globStep != 0:
            return

        # 新模型，我们加载预先训练的word2vec数据并初始化嵌入
        embeddings_path = os.path.join(self.args.rootDir, 'data', 'embeddings', self.args.embeddingSource)
        embeddings_format = os.path.splitext(embeddings_path)[1][1:]
        print("Loading pre-trained word embeddings from %s " % embeddings_path)
        with open(embeddings_path, "rb") as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vector_size
            initW = np.random.uniform(-0.25, 0.25, (len(self.textData.word2id), vector_size))
            for line in tqdm(range(vocab_size)):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode('utf-8')
                        break
                    if ch != b'\n':
                        word.append(ch)
                if word in self.textData.word2id:
                    if embeddings_format == 'bin':
                        vector = np.fromstring(f.read(binary_len), dtype='float32')
                    elif embeddings_format == 'vec':
                        vector = np.fromstring(f.readline(), sep=' ', dtype='float32')
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)
                    initW[self.textData.word2id[word]] = vector
                else:
                    if embeddings_format == 'bin':
                        f.read(binary_len)
                    elif embeddings_format == 'vec':
                        f.readline()
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)

        # PCA分解可减少word2vec维数
        if self.args.embeddingSize < vector_size:
            U, s, Vt = np.linalg.svd(initW, full_matrices=False)
            S = np.zeros((vector_size, vector_size), dtype=complex)
            S[:vector_size, :vector_size] = np.diag(s)
            initW = np.dot(U[:, :self.args.embeddingSize], S[:self.args.embeddingSize, :self.args.embeddingSize])

        # 初始化输入和输出嵌入
        sess.run(em_in.assign(initW))
        sess.run(em_out.assign(initW))

    def managePreviousModel(self, sess):
        """
        根据参数还原或重置模型
        如果目标目录已经包含一些文件，它将按以下方式处理冲突：
         *如果设置了--reset，将删除所有当前文件（警告：不要求确认）并进行训练 从头开始（globStep和cie重新初始化）
         *否则，将取决于目录内容。如果目录包含：
           *没有模型文件（仅摘要日志）：用作重置（从头开始）
           *其他模型文件，但未找到modelName（肯定已更改keepAll选项）：引发错误，用户应自己决定做什么
           *正确的模型文件（最终是其他文件）：没问题，只需恢复培训
        无论如何，该目录都将由summary writer创建
        @param sess: 当前正在运行的会话
        """

        print('WARNING: ', end='')

        modelName = self._getModelName()

        if os.listdir(self.modelDir):
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.modelDir))
            # 分析目录内容
            elif os.path.exists(modelName):  # Restore the model
                print('Restoring previous model from {}'.format(modelName))
                self.saver.restore(sess,
                                   modelName)  # Will crash when --reset is not activated and the model has not been saved yet
            elif self._getModelList():
                print('Conflict with previous models.')
                raise RuntimeError(
                    'Some models are already present in \'{}\'. You should check them first (or re-try with the keepAll flag)'.format(
                        self.modelDir))
            else:  # 没有其他模型可与之冲突（可能是摘要文件）
                print('No previous model found, but some files found at {}. Cleaning...'.format(
                    self.modelDir))  # Warning: No confirmation asked
                self.args.reset = True

            if self.args.reset:
                fileList = [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir)]
                for f in fileList:
                    print('Removing {}'.format(f))
                    os.remove(f)

        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))

    def _saveSession(self, sess):
        """
        保存模型参数和变量
        @param sess: 当前的session
        """
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        self.saveModelParams()
        model_name = self._getModelName()
        with open(model_name, 'w') as f:  # HACK: Simulate the old model existance to avoid rewriting the file parser
            f.write('This file is used internally by DeepQA to check the model existance. Please do not remove.\n')
        self.saver.save(sess, model_name)  # TODO: Put a limit size (ex: 3GB for the modelDir)
        tqdm.write('Model saved.')

    def _getModelList(self):
        """
        返回模型目录中的模型文件列表
        """
        return [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir) if f.endswith(self.MODEL_EXT)]

    def loadModelParams(self):
        """
        加载与当前模型关联的一些值，例如当前globStep值
        目前，在加载模型之前无需调用此函数（不恢复任何参数）。然而，
        modelDir名称将在此处初始化，因此需要在managePreviousModel（）_getModelName（）或_getSummaryName（）之前调用此函数，
        警告：如果您修改此功能，请确保所做的更改反映了saveModelParams，还请检查参数是否应该在managePreviousModel中重置
        """
        # 获取当前模型路径
        self.modelDir = os.path.join(self.args.rootDir, self.MODEL_DIR_BASE)
        if self.args.modelTag:
            self.modelDir += '-' + self.args.modelTag

        # 如果存在以前的模型，恢复一些参数
        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if not self.args.reset and not self.args.createDataset and os.path.exists(configName):
            # 加载ing
            config = configparser.ConfigParser()
            config.read(configName)

            # 检查版本
            currentVersion = config['General'].get('version')
            if currentVersion != self.CONFIG_VERSION:
                raise UserWarning(
                    'Present configuration version {0} does not match {1}. You can try manual changes on \'{2}\''.format(
                        currentVersion, self.CONFIG_VERSION, configName))

            # 恢复参数
            self.globStep = config['General'].getint('globStep')
            self.args.watsonMode = config['General'].getboolean('watsonMode')
            self.args.autoEncode = config['General'].getboolean('autoEncode')
            self.args.corpus = config['General'].get('corpus')

            self.args.datasetTag = config['Dataset'].get('datasetTag')
            self.args.maxLength = config['Dataset'].getint(
                'maxLength')  # We need to restore the model length because of the textData associated and the vocabulary size (TODO: Compatibility mode between different maxLength)
            self.args.filterVocab = config['Dataset'].getint('filterVocab')
            self.args.skipLines = config['Dataset'].getboolean('skipLines')
            self.args.vocabularySize = config['Dataset'].getint('vocabularySize')

            self.args.hiddenSize = config['Network'].getint('hiddenSize')
            self.args.numLayers = config['Network'].getint('numLayers')
            self.args.softmaxSamples = config['Network'].getint('softmaxSamples')
            self.args.initEmbeddings = config['Network'].getboolean('initEmbeddings')
            self.args.embeddingSize = config['Network'].getint('embeddingSize')
            self.args.embeddingSource = config['Network'].get('embeddingSource')

            # 无需恢复训练参数，批次大小或其他与模型无关的参数

            # 显示恢复的参数
            print()
            print('Warning: Restoring parameters:')
            print('globStep: {}'.format(self.globStep))
            print('watsonMode: {}'.format(self.args.watsonMode))
            print('autoEncode: {}'.format(self.args.autoEncode))
            print('corpus: {}'.format(self.args.corpus))
            print('datasetTag: {}'.format(self.args.datasetTag))
            print('maxLength: {}'.format(self.args.maxLength))
            print('filterVocab: {}'.format(self.args.filterVocab))
            print('skipLines: {}'.format(self.args.skipLines))
            print('vocabularySize: {}'.format(self.args.vocabularySize))
            print('hiddenSize: {}'.format(self.args.hiddenSize))
            print('numLayers: {}'.format(self.args.numLayers))
            print('softmaxSamples: {}'.format(self.args.softmaxSamples))
            print('initEmbeddings: {}'.format(self.args.initEmbeddings))
            print('embeddingSize: {}'.format(self.args.embeddingSize))
            print('embeddingSource: {}'.format(self.args.embeddingSource))
            print()

        # 现在，编码器和解码器之间没有任意独立的maxLength
        self.args.maxLengthEnco = self.args.maxLength
        self.args.maxLengthDeco = self.args.maxLength + 2

        if self.args.watsonMode:
            self.SENTENCES_PREFIX.reverse()

    def saveModelParams(self):
        """
        保存模型参数，例如当前的globStep值
        警告：如果您修改此功能，请确保映射到了loadModelParams
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version'] = self.CONFIG_VERSION
        config['General']['globStep'] = str(self.globStep)
        config['General']['watsonMode'] = str(self.args.watsonMode)
        config['General']['autoEncode'] = str(self.args.autoEncode)
        config['General']['corpus'] = str(self.args.corpus)

        config['Dataset'] = {}
        config['Dataset']['datasetTag'] = str(self.args.datasetTag)
        config['Dataset']['maxLength'] = str(self.args.maxLength)
        config['Dataset']['filterVocab'] = str(self.args.filterVocab)
        config['Dataset']['skipLines'] = str(self.args.skipLines)
        config['Dataset']['vocabularySize'] = str(self.args.vocabularySize)

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args.hiddenSize)
        config['Network']['numLayers'] = str(self.args.numLayers)
        config['Network']['softmaxSamples'] = str(self.args.softmaxSamples)
        config['Network']['initEmbeddings'] = str(self.args.initEmbeddings)
        config['Network']['embeddingSize'] = str(self.args.embeddingSize)
        config['Network']['embeddingSource'] = str(self.args.embeddingSource)

        # 跟踪学习参数（但不还原它们）
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args.learningRate)
        config['Training (won\'t be restored)']['batchSize'] = str(self.args.batchSize)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)

        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

    def _getSummaryName(self):
        """
        解析参数以将summary保存在与模型相同的位置
        如果我们恢复训练，该文件夹可能已经包含日志，这些日志将被合并
        @return: summary的地址和名字
        """
        return self.modelDir

    def _getModelName(self):
        """
        解析参数以决定要保存或者是加载模型
        在每个检查点以及首次加载模型时都会调用此函数。如果设置了keepAll选项，则
        globStep值将包含在名称中。
        @return: string: 需要保存的模型的路径和名称
        """
        modelName = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        if self.args.keepAll:  # We do not erase the previously saved model by including the current step on the name
            modelName += '-' + str(self.globStep)
        return modelName + self.MODEL_EXT

    def getDevice(self):
        """
        解析参数以确定在哪个设备上运行模型
        @return: string:运行程序的设备的名称
        """
        if self.args.device == 'cpu':
            return '/cpu:0'
        elif self.args.device == 'gpu':
            return '/gpu:0'
        elif self.args.device is None:  # No specified device (default)
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
