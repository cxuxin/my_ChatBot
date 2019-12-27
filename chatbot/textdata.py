#!/usr/bin/env python
# encoding: utf-8
# @Software: PyCharm
# @Project: Chatbot_Seq2Seq
# @File: textdata.py
# @Desc:加载对话语料库，构建词汇
import collections
import os
import pickle
import random
import string

import nltk
import numpy as np
from tqdm import tqdm

from chatbot.corpus.cornelldata import CornellData



class Batch:
    """
    存放batch信息的类
    """

    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    """"
    数据集类
    """

    availableCorpus = collections.OrderedDict([  # OrderedDict，因为第一个元素是默认选择
        ('cornell', CornellData),

    ])

    @staticmethod
    def corpusChoices():
        """
        返回可用的数据集
        @return: list<string>: 支持的语料库
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """
        加载所有conversations
        @param args: 模型的参数
        """
        # 模型参数
        self.args = args

        # 路径参数
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )  # Sentences/vocab filtered for this model

        self.padToken = -1  # Padding
        self.goToken = -1  # 序列开始符
        self.eosToken = -1  # 序列结束符
        self.unknownToken = -1  # 从词汇表中删除的词

        self.trainingSamples = []  # 包含每个问题及其答案的2d数组[[input,target]]

        self.word2id = {}
        self.id2word = {}
        self.idCount = {}

        self.loadCorpus()

        self._printStats()

        if self.args.playDataset:
            self.playDataset()

    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

    def _constructBasePath(self):
        """
        @return: 返回当前数据集的基本前缀的名称
        """
        path = os.path.join(self.args.rootDir, 'data' + os.sep + 'samples' + os.sep)
        path += 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        return path

    def makeLighter(self, ratioDataset):
        """
        只保留数据集的一小部分，由比率给出
        """
        # if not math.isclose(ratioDataset, 1.0):
        #    self.shuffle()  # Really ?
        #    print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """
        打乱训练样本
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)

    def _createBatch(self, samples):
        """
        @param samples: (list<Obj>):一个样本列表，每个样本都在表单[input, target]中。
        @return:一个Batch对象
        """

        batch = Batch()
        batchSize = len(samples)

        # 创建batch张量
        for i in range(batchSize):
            sample = samples[i]
            if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
                sample = list(reversed(sample))
            if not self.args.test and self.args.autoEncode:  # 自动编码:输入和输出都使用问题或答案
                k = random.randint(0, 1)
                sample = (sample[k], sample[k])

            batch.encoderSeqs.append(list(reversed(sample[0])))  # 反向输入(而不是输出)，在原来的seq2seq论文中定义的小技巧
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:])  # 与解码器相同，但向左移动(忽略<go>)

            # 长句应该在数据集创建期间被过滤
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # Add padding & define weight
            batch.encoderSeqs[i] = [self.padToken] * (self.args.maxLengthEnco - len(batch.encoderSeqs[i])) + \
                                   batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append(
                [1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (
                        self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.padToken] * (
                        self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        return batch

    def getBatches(self):
        """
        为当前epoch准备batch
        @return: list<Batch>:获取下一个epoch的batch列表
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """
            生成器上面的小批量训练样本
            """
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]

        # TODO: Should replace that by generator (better: by tf.queue)

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    def getSampleSize(self):
        """
        返回数据集的大小
        @return: int: 训练样本的数量
        """
        return len(self.trainingSamples)

    def getVocabularySize(self):
        """
        返回数据集中出现的单词数
        @return: int:加载程序语料库上的单词数
        """
        return len(self.word2id)

    def loadCorpus(self):
        """
        加载/创建对话数据
        """
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:  # 第一次加载数据库:创建所有文件
            print('Training samples not found. Creating dataset...')

            datasetExist = os.path.isfile(self.fullSamplesPath)  # 尝试从预处理的条目构造数据集
            if not datasetExist:
                print('Constructing full dataset...')

                optional = ''

                # Corpus creation
                corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir + optional)
                self.createFullCorpus(corpusData.getConversations())
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()

            print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
                self.args.vocabularySize,
                self.args.filterVocab
            ))
            self.filterFromFull()  # Extract the sub vocabulary for the given maxLength and filterVocab

            # Saving
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath)  # Saving tf samples
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0

    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # 警告:如果在这里添加内容，也要修改loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainingSamples': self.trainingSamples
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """
        Load samples from file
        @param filename: (str) pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # 警告:如果在这里添加一些东西，也要修改saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words

    def filterFromFull(self):
        """
        加载预处理的完整语料库并过滤词汇/句子以匹配给定的模型选项
        """

        def mergeSentences(sentences, fromEnd=False):
            """
            合并这些句子，直到达到最大的句子长度 另外，对于未使用的句子，减少id计数
            @param sentences: (list<list<int>>):当前行的句子列表
            @param fromEnd:  在答案上定义问题
            @return: 句子中单词id的列表
            """
            # 一句一句地加，直到达到最大的长度
            merged = []

            # 如果是问题：我们只保留最后一句
            # 如果是答案：我们只保留第一句话
            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                # 如果总长度不是太大，我们还可以再加一个句子
                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd:
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:
                    for w in sentence:
                        self.idCount[w] -= 1
            return merged

        newSamples = []

        # 第一步:遍历所有单词，根据句子长度添加过滤语句
        for inputWords, targetWords in tqdm(self.trainingSamples, desc='Filter sentences:', leave=False):
            inputWords = mergeSentences(inputWords, fromEnd=True)
            targetWords = mergeSentences(targetWords, fromEnd=False)

            newSamples.append([inputWords, targetWords])
        words = []

        # 第二步:过滤未使用的单词，并用未知的标记替换它们
        # 这也是我们更新correnspondance字典的地方
        specialTokens = {
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        newMapping = {}  # 将完整的单词id映射到新的id
        newId = 0

        selectedWordIds = collections \
            .Counter(self.idCount) \
            .most_common(self.args.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.args.filterVocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  # Iterate in order
            if wordId in selectedWordIds:
                newMapping[wordId] = newId
                word = self.id2word[wordId]
                del self.id2word[wordId]
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else:
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]
                del self.id2word[wordId]

        # 最后一步:用新id替换旧id，过滤空句子
        def replace_words(words):
            valid = False  # 过滤空序列
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  # 如果只包含未知的标记，也要过滤
                    valid = True
            return valid

        self.trainingSamples.clear()

        for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            valid &= targetWords.count(self.unknownToken) == 0  # Filter target with out-of-vocabulary target words ?

            if valid:
                self.trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple

        self.idCount.clear()

    def createFullCorpus(self, conversations):
        """
        从给定的词汇表中提取所有数据。
        将数据保存在磁盘上。请注意，整个语料库都是预先处理过的，没有句子长度或词汇量的限制。
        """
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary

        # Preprocessing data
        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

    def extractConversation(self, conversation):
        """
        从对话中提取samples
        @param conversation: 包含要提取的行的对话对象
        """

        if self.args.skipLines:  # WARNING: The dataset won't be regenerated if the choice evolve (have to use the datasetTag)
            step = 2
        else:
            step = 1

        for i in tqdm_wrap(
                range(0, len(conversation['lines']) - 1, step),  # 忽略最后一行(没有answer)
                desc='Conversation',
                leave=False
        ):
            inputLine = conversation['lines'][i]
            targetLine = conversation['lines'][i + 1]

            inputWords = self.extractText(inputLine['text'])
            targetWords = self.extractText(targetLine['text'])

            if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
                self.trainingSamples.append([inputWords, targetWords])

    def extractText(self, line):
        """
        从示例行中提取单词
        @param line: (str):包含要提取的文本的line
        @return: list<list<int>>:句子中单词id的句子list
        """
        sentences = []  # List[List[str]]

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])

            tempWords = []
            for token in tokens:
                tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

            sentences.append(tempWords)

        return sentences

    def getWordId(self, word, create=True):
        """
        获取单词的id(如果不存在，则将其添加到字典中)。如果单词不存在，且create被设置为False，那么该函数将返回unknownToken值
        @param word:  (str):要添加的单词
        @param create: (Bool):如果是真，并且这个单词还不存在，那么将添加word
        @return: 创建的单词的id
        """

        word = word.lower()  # 忽略大小写

        # 在推理时，我们只是简单地查找这个词
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # 如果单词已经存在，则获取id
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # 如果没有，则创建一个新条目
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def printBatch(self, batch):
        """
        打印完整的Batch，对调试很有用
        @param batch: a batch object
        @return:
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(
                ' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """
        将sequence转换为人类可读的字符串
        @param sequence: (list<int>)要打印的sequence
        @param clean: (Bool)如果设置了，则删除<go>、<pad>和<eos>tokens
        @param reverse: (Bool)对于输入，可选择恢复标准顺序
        @return: str: 句子
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """
        与空格连接的更简洁的版本
        @param tokens:(list<string>): 要打印的句子
        @return:句子
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
            else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """
        将整数列表转换为人类可读的字符串。
        前一个函数的不同之处在于，在batch对象上，值被重新组织为batch而不是sentence。
        @param batchSeq: (list<list<int>>):打印的句子
        @param seqId: 序列在batch中的位置
        @param kwargs: 格式化选项(参见sequence2str())
        @return:
        """
        sequence = []
        for i in range(len(batchSeq)):  # 序列长度
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """
        编码一个序列并返回一个batch作为模型的输入
        @return: 包含句子的batch对象，如果出错则为none
        """

        if sentence == '':
            return None

        # 第一步:把句子用记号分开
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # 第二步:转换单词id中的token
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # 第三步:创建batch(add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """
        译码器的输出，并返回一个人类友好的句子
        @param decoderOutputs: (list<np.array>)
        @return:
        """
        sequence = []

        # 选择预测得分最高的单词
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # 添加每个预测的单词id

        return sequence  # 我们返回原始语句。让调用者最后做一些清理工作

    def playDataset(self):
        """
        从数据集中打印一个随机对话
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
