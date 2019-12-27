#!/usr/bin/env python
# encoding: utf-8
# @Software: PyCharm
# @Project: Chatbot_Seq2Seq
# @File: model.py
# @Desc: 在给定输入序列的情况下预测下一个句子的模型
import tensorflow as tf


class ProjectionOp:
    """
    单层感知器
    在输出尺寸上投影输入张量
    """

    def __init__(self, shape, scope=None, dtype=None):
        """
        @param shape(tuple): 输入dim，输出dim
        @param scope(str): 封装变量
        @param dtype: 权重类型
        """
        assert len(shape) == 2
        self.scope = scope
        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        return self.W, self.b

    def __call__(self, X):
        """
        将解码器的输出投影到词汇空间中
        @param X: X（tf.Tensor）：输入值
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    seq2seq模型的实现
    构成：
        编码器Encoder/解码器decoder
        2个LTSM层
    """

    def __init__(self, args, textData):
        """
        @param args: 模型参数
        @param textData: 数据集类
        """
        print("Model creation...")

        self.textData = textData  # 保留数据集副本
        self.args = args  # 跟踪模型的参数
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs = None
        self.decoderInputs = None
        self.decoderTargets = None
        self.decoderWeights = None

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # 创建网络结构
        self.buildNetwork()

    def buildNetwork(self):
        """
        创建网络结构
        """

        # 采样softmax的参数（需要注意机制和较大的词汇量）
        outputProjection = None

        # 仅当我们采样的词汇量小于词汇量时，softmax采样才有意义。
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # 加一维

                # 使用32位浮点数来计算sampled_softmax_loss以避免数值不稳定。
                localWt = tf.cast(outputProjection.W_t, tf.float32)
                localB = tf.cast(outputProjection.b, tf.float32)
                localInputs = tf.cast(inputs, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # 应具有形状[num_classes，dim]
                        localB,
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  # 每个batch要随机采样的class数
                        self.textData.getVocabularySize()),  # class的个数
                    self.dtype)

        # 1)构建RNN cell:构建Seq2Seq深度神经网络中的Encoder和Decoder中的RNN Cell，可以采用LSTM实现
        def create_rnn_cell():
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                self.args.hiddenSize,
            )
            if not self.args.test:
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout
                )
            return encoDecoCell

        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # 2)构建深度神经网络输入占位符
        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in
                                  range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in
                                  range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in
                                   range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in
                                   range(self.args.maxLengthDeco)]

        # 3)构建Seq2Seq模型，其中的Cell使用的是第1步构建的Cell
        # 此处使用Embedding模型，输入的句子中的词Id会转换为向量表示，这样就能更好地表示单词
        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # 为了进行训练，我们强制输出正确的内容（feed_previous = False）
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  # 编码器和解码器具有相同的class数
            embedding_size=self.args.embeddingSize,  # 每个词的维数
            output_projection=outputProjection.getWeights() if outputProjection else None,
            feed_previous=bool(self.args.test)  # 当测试（self.args.test）时，将上一个输出用作下一个输入（feed_previous）
        )

        # 训练并减少内存使用量。其他解决方案，使用softmax采样

        # 仅用于测试
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]

            # TODO: Attach a summary to visualize the output

        # 仅用于训练
        else:
            # 4) 定义损失函数，选用sequence_loss作为损失函数
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function=sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # 跟踪cost

            # 5)选择优化器，选择Adam优化算法进行训练
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):
        """
        前进/训练step
        @param batch: 在testing mode下输入数据，在output mode下输入和目标
        @return: dict:
        """

        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
