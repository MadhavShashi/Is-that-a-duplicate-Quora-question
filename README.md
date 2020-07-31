![img](https://user-images.githubusercontent.com/49862149/88937413-bff08500-d2a1-11ea-9dce-5a1d21d8a47d.png)
## Overview
<p> Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.<p>
<p> Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term. <p>

>Credits: Kaggle

## Motivation
What could be a perfect way to utilize unfortunate lockdown period? The only solutions to handle the situation are definitely among one of the smart ways to utilize the time industriously. Like most of you, I spend my time in YouTube, Netflix, coding and reading some research papers on weekends. The idea of classifying “Is that a duplicate Quora question?” struck to me when I was browsing through some research papers. Specially, when I found a You Tube video of Kaggle grandmaster “Abhishek Thakur” about this topic. I find a relevant research paper associated with it. And that led me to collect the Dataset of “Is that a duplicate Quora question?”  to train a Machine learning model.

## Sources/Useful Links
- Video Link : https://www.youtube.com/watch?v=vA1V8A69e9c
- SlideShare Link : https://www.slideshare.net/abhishekkrthakur/is-that-a-duplicate-quora-question
- Blog 1 : https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0&preview=2761178.pdf
- Blog 2 : https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0&preview=NilssonTiman.pdf

## Problem Statement
-	Identify which questions asked on Quora are duplicates of questions that have already been asked.
- This could be useful to instantly provide answers to questions that have already been answered.
- We are tasked with predicting whether a pair of questions are duplicates or not.

## Solution
Suppose we have a fairly large data set of question-pairs that has been labeled (by humans) as **“duplicate” or “not duplicate.”** We could then use natural language processing *(NLP)* techniques to extract the difference in meaning or intent of each question-pair, use machine learning (ML) to learn from the human-labeled data, and predict whether a new pair of questions is duplicate or not.

## Which type of ML Problem is this?
![#FF5733](https://via.placeholder.com/7x24/FF5733/000000?text=+) It is a **binary classification problem**, for a given pair of questions we need to predict if they are *duplicate or not.*   

## What is the best performance metric for this Problem?
- **log-loss:**     https://www.kaggle.com/wiki/LogarithmicLoss
   * **Qns:** *Why log-loss is right Metric for this??*<p>**Ans:** *This is a **“Binary class classification problem”** this doesn’t mean we want output as “**0**” or “**1**”. we want “ **p (q1 ≈ q2)** “ and here probability lies b/w “**0 to 1**”, and when we have probability value and predicting for binary class classification problem* **the log-loss is one of the best metric**.</p>
- **Binary Confusion Matrix**

## Business Objectives and Constraints
1.  **The cost of a mis-classification can be very high.**
2.  **You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.**
    * **Qsn:** *Why we choose any threshold of choice??*<p>**Ans:** *This mean, see we want **“p (q1 ≈ q2)“** and here probability lies b/w **“ 0 to 1”**, so here we can choose some threshold which confirm me* **“ q1 ≈ q2 ”**.</p>
    * **Example**: If we choose threshold **0.95**, this mean **p(q1 ≈ q2)** when **p>0.95**.
    * **Benefit of choosing threshold here**: If suppose we set threshold >0.95 and Human read the answer and they told this is the wrong answer for this question, then we can change the threshold. 
3.	**No strict latency concerns.**
4.	**Interpretability is partially important.**

## Data Overview
- Data will be in a file *Train.csv*
- Train.csv contains *5 columns*: qid1, qid2, question1, question2, is_duplicate
- Number of rows in Train.csv = **404,290**
#### Example Data point 
| id | qid1 | qid2 | question1 | question2 | is_duplicate |
|--- | --- | --- | ------ | -------- | --- |
| 0 | 1 | 2 | What is the step by step guide to invest in share market in India? | What is the step by step guide to invest in share market? | 0 |
| 1 | 3 | 4 | What is the story of Kohinoor (Koh-i-Noor) Diamond? | What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back? | 0 |

## Train and Test ratio
![#FF5733](https://via.placeholder.com/7x24/FF5733/000000?text=+) We build train and test by *randomly splitting* in the ratio of **60:40** or **70:30** whatever we choose as we have sufficient points to work with.

## Agenda

### 1. Analyzing the Data (EDA)



```python
fuzz.ratio("YANKEES", "NEW YORK YANKEES") ⇒ 60
fuzz.ratio("NEW YORK METS", "NEW YORK YANKEES") ⇒ 75
```

| Data Size | Model Name | Features | Tuning | Log Loss |
|---------- | ---------- | -------- | ------ | -------- |
| ~ 404K | Random | Sim Fs+Adv Fs+TFIDF Weighted W2V | NA | **0.88** |
| Data Size | Model Name | Features | Tuning | Log Loss |
