#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:39:05 2017

@author: steveoni
"""
from __future__ import division
import re
import math
import numpy as np


class Bayes(object):
    
    def __init__(self):
        self.localstorage = {'_Bayes::registeredLabels':''}#serve as the database
        self.label = [] #collect all label
        self.lab =[] #temporary stored label
    
    def __unique(self,a):
        """
        find the unique character in asentance and turn into a list
        """
         ar = set(a)
         return list(ar)
     
    def __stemKey(self,stem,label):
        """
        create key for each character and store them with their respective
        label name e.g for word 'hen' with label 'english' is stored as
        '_Bayes::stem:hen::label:english'
        """
        return '_Bayes::stem:' + stem + '::label:' + label
    
    def __docCountKey(self,label):
        """
        same as __stemkey but only store label document
        """
        return '_Bayes::docCount:' + label
    
    def __stemCountKey(self,stem):
        return '_Bayes::stemCount:' + stem
    
    def __tokenize(self,text):
        """
        tokenize word and find the unique character
        """
        text = text.lower()
        text = re.sub('/\W/g',' ',text)
        text = re.sub('/\s+/g',' ',text)
        text = text.strip()
        text = text.split(' ')
        text = self.__unique(text)
        return text
    
    def getLabels(self):
        """
        get the label and turn them to list
        """"
        key ='_Bayes::registeredLabels' 
        
        
        if key in self.localstorage.keys():
            label = self.localstorage['_Bayes::registeredLabels']
        else:
            label=''
        
           
        label = label.split(',')
        
        label = filter(lambda x:len(x),label)#make it return empty list if no label found
        
        self.lab +=label
        return self.lab
        
        
    
    def __registerLabel(self,label):
        """
        check if label exist if not registered the label
        """
        labels = self.getLabels()
        
        if label not in labels:
            labels.append(label)
            
            for i in labels:
                if i not in self.label:
                    self.label.append(label)
                    self.localstorage['_Bayes::registeredLabels'] = ','.join(self.label)
            
            
        
        return labels
    
    def __stemLabelCount(self,stem,label):
        """
        count the number of time a word with a particular label occur
        """
        count = 0
        key =self.__stemKey(stem,label) 
        if key in self.localstorage.keys():
            count  = int(self.localstorage[key])
        else:
            count =0
        
            
        return count
    
    def __stemInverseLabelCount(self,stem,label):
        """
        count the number of time a word does not occur
        """
        labels = self.getLabels()
        total = 0
        
        for i in range(len(labels)):
            
            if labels[i] == label:
                continue
                
            total += int(self.__stemLabelCount(stem,labels[i]))
            
        return total
    
    def __stemTotalCount(self,stem):
        count = 0
        key  = self.__stemCountKey(stem)
        if key in self.localstorage.keys():
            count = int(self.localstorage[key])
        else:
            count = 0
        
        return count
    
    def __docCount(self,label):
        """
        count how many document a label have
        """
        count = 0
        key = self.__docCountKey(label)
        if key in self.localstorage.keys():
            count = int(self.localstorage[key])
        else:
            count =0
            
        return count
    
    def __docInverseCount(self,label):
        labels =self.getLabels()
        total =0
        
        for i in range(len(labels)):
            if labels[i] == label:
                continue
                
            total +=int(self.__docCount(labels[i]))
            
        return total
    
    def __increment(self,key):
        """
        check if key and word are seen before
        if seen increment the count
        """
        count = 0
        if key in self.localstorage.keys():
            count = int(self.localstorage[key])
        else:
            count = 0
            
        self.localstorage[key] =count +1
        
        return count + 1
    
    def __incrementStem(self,stem,label):
        self.__increment(self.__stemCountKey(stem))
        self.__increment(self.__stemKey(stem,label))
        
    def __incrementDocCount(self,label):
        return self.__increment(self.__docCountKey(label))
    
    
        
    def train(self,text,label):
        """
        train the label
        text: document to be trained
        label:the document label
        we rigistered the the label and store them in the localstorage
        if seen before it is incremented
        """
        self.__registerLabel(label)
        words = self.__tokenize(text)
        length = len(words)
        for i in range(length):
            self.__incrementStem(words[i],label)
            
        self.__incrementDocCount(label)
        
    def guess(self,text):
        """
        take a document and geuss the label
        """
        words = self.__tokenize(text)
        length = len(words)
        labels = self.getLabels()
        totalDocCount =0
        docCounts = {}
        docInverseCounts = {}
        scores ={}
        labelProbability = {}
        #wordicity =0
        
        for j in range(len(labels)):
            label = labels[j]
            docCounts[label] = self.__docCount(label)
            docInverseCounts[label] = self.__docInverseCount(label)
            totalDocCount +=int(docCounts[label])
        for j in range(len(labels)):
            label = labels[j]
            logsum =0
            labelProbability[label] = (docCounts[label] / totalDocCount) if totalDocCount!=0 else 0 #to prevent zerodivision error
            
            for i in range(length):
                word = words[i]
                _stemTotalCount = self.__stemTotalCount(word)
                if _stemTotalCount ==0:
                    continue
                else:
                    
                    wordProbability = self.__stemLabelCount(word,label)
                    wordInverseProbability =self.__stemInverseLabelCount(word,label) / docInverseCounts[label]
                    wordicity = wordProbability / (wordProbability + wordInverseProbability)#bayes p{A|B}=P(A)P{B|A}/(P(A)P{B|A}+P(A')P{B|A'})
                    
                    wordicity =((1 * 0.5) + (_stemTotalCount * wordicity))/(1 + _stemTotalCount)#the '1' is the weight it should have been an input
            
                    if wordicity ==0:
                        wordicity = 0.01
                    elif wordicity ==1:
                        wordicity =0.99 
                    
                logsum += (np.log(1 - wordicity)-np.log(wordicity))#prevent overflow of floating number
                print label+ 'icity of '+ word +':',wordicity
            scores[label] = 1 /(1+ np.exp(logsum))
        return scores
    def extractWinner(self,scores):
        """
        return the label with the best score
        """
        bestscore = 0
        bestLabel = None
        for label in scores:
            if scores[label] > bestscore:
                bestscore = scores[label]
                bestLabel = label
                
        return {'label':bestLabel,'score':bestscore}

    
