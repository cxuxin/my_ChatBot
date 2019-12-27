# Chatbot_Seq2Seq

基于Github上[DeepQA](https://github.com/Conchylicultor/DeepQA)项目进行修改

#### 目录

* [介绍](#介绍)
* [安装说明](#安装说明)
* [运行](#运行)
* [结果](#结果)

#### 介绍
该项目基于论文[A Neural Conversational Model](http://arxiv.org/abs/1506.05869)  
利用RNN (seq2seq模型)进行句子预测，基于python和TensorFlow完成。

该项目的预料库使用[Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).已包含在项目文件下。


#### 安装
该程序需要以下依赖:
 * python 3.5+
 * tensorflow (在v1.14.0下完成测试)
 * numpy
 * nltk (用于标记句子的自然语言工具包)
 * tqdm (进度条工具包)
 * CUDA (使用GPU训练，可选)

您可能还需要下载其他数据以使nltk生效。  
```
>>>import nltk
>>>nltk.download()
```

#### 运行

要训练模型，只需运行 `python main.py`.训练完成后，可以运行`python main.py --test`
(在`save/model/samples_predictions.txt`中生成结果)或 `main.py—test interactive`进行实时提问测试。

要使用[TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)可视化计算图形和成本，
只需运行`TensorBoard—logdir save/`

默认情况下，网络架构是一个标准的编码器/解码器：  
具有两个LSTM层(隐藏大小为`512`)和一个嵌入大小为`64`的词汇表。网络使用`ADAM`进行训练。
句子的最大长度设置为`10`个单词，但是可以增加。

#### 结果
在经过3个小时左右的时间训练后可以得到模型结果：
    
    Q: Hello!
    A: Hello.

    Q: Hi!
    A: Hi.
    
    Q: you are a rebort, right?
    A: I'm not.

    Q: you are not a rebort?
    A: No, i'm not.

    Q: How are you?
    A: Fine.

    Q: What is your name?
    A: Bedevere, my liege.

    Q: How old are you
    A: Fifteen.

    Q: Who is John?
    A: It's me.
    
    Q: are you John?
    A: Yes.
    
    Q: hello,John!
    A: Hello.
    
    Q: Can a cat walk ?
    A: Sure. why not?
    
    Q: Can a fish walk ?
    A: Absolutely not.
    
    Q: Tell me a story
    A: I'm sorry.
    
    Q: Once upon a time...
    A: Yeah, that's right.
    
    Q: what do you like to talk about ?
    A: I do n't know.
    
    Q: You are so nice!
    A: I'm not.

    Q: See you later.
    A: Bye.

以上为回答较为准确的样例，还有一些采用安全回答（如“i don't know. i'm sorry”）
等无意义答案进行回答，更多样例请看文件`save/model/samples_predictions.txt`
