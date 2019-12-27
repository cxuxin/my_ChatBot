#!/usr/bin/env python
# encoding: utf-8
# @Software: PyCharm
# @Project: Chatbot_Seq2Seq
# @File: cornelldata.py
# @Desc:加载cornell对话语料库。
# 语料库可在以下链接下载:
# http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
import ast
import os


class CornellData:
    def __init__(self, dirName):
        """
        @param dirname(string): 语料库地址
        """
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

        # 读取对话句子
        self.lines = self.loadLines(os.path.join(dirName, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        # 读取每轮对话
        self.conversations = self.loadConversations(os.path.join(dirName, "movie_conversations.txt"),
                                                    MOVIE_CONVERSATIONS_FIELDS)

    def loadLines(self, fileName, fields):
        """
        @param filename: (string)加载的文件
        @param fields: (set<string>)文件内容格式
        @return: (dict<dict<str>>)每行提取的字段
        """
        lines = {}

        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                # 提取字段
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self, fileName, fields):
        """
        @param filename: (string)加载的文件
        @param fields: (set<string>)文件内容格式
        @return: (list<dict<string>>)每行提取的字段
        """
        conversations = []

        with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
            for line in f:
                values = line.split(" +++$+++ ")

                # 提取字段
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                # 将字符串转化为列表
                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                # 给conv_obj添加属性值lines 存储具体对话的内容
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])

                conversations.append(convObj)

        return conversations

    def getConversations(self):
        return self.conversations
