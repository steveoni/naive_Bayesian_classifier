## naive Bayesian classifier


The classifier is good at classifing document to a specify label ,e.g if to
detect spam and ham, it is trained with dataset containing spam and ham class

In the example we trained the classifier on english, spanish and french so that it can determine if an unknow document is french,english or spanish

The tutorial is a re-implementation of this javascript program [naive Bayes in javascript](https://www.burakkanber.com/blog/machine-learning-naive-bayes-1) ,you can visit the site for better understanding of the algorithm

```python
Bayes= Bayes()

Bayes.train("the killed man",'ham')
Bayes.train("animal of gangazia",'spam')

scores = Bayes.geuss("the man of gag")

Bayes.extractwinner(scores)
#{score:0.95,label:'spam'}
```
