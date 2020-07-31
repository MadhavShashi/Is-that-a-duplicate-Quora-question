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
- Some Analysis on Train Data Set below:
- Getting Deep knowledge of Data Set (on question parameter)
  * Output:
    * ```(1). Total number of questation pairs for training:- 404290```
    * ```(2). Questation pairs are not similar (is_duplicate= 0) in percentage:- 63.08%```
    * ```(3). Questation pairs are similar (is_duplicate= 1) in percentage:- 36.92%```
  
  * Plotted above detail’s on graph:
      ```python
        df_train.groupby("is_duplicate")["id"].count().plot.bar()
      ```
      ![p1](https://user-images.githubusercontent.com/49862149/89006795-1b178b80-d325-11ea-91b7-272c7a7c03ea.png) 
      
      We can clearly see this graph and analyze it, positive class (**is_duplicate=0**) has more pair question than negative class (**is_duplicate=1**). We can think this as unbalanced data set.
- *Now*, Getting Deep knowledge about Number of __unique questions__:
  
  * Output:
    * ```(1). Total number of Unique Questions are: - 537933```
    * ```(2). Number of unique questions that appear more than one time: - 111780 (20.7%)```
    * ```(3). Max number of times a single question is repeated:- 157```
  * Plotting Number of occurrences of each question:
    ![p2](https://user-images.githubusercontent.com/49862149/89007871-3d120d80-d327-11ea-9b5d-76b6ea5a6871.png)
    
    In terms of questions, most questions only appear a few times, with very few questions appearing several times (and a few questions appearing many times). One question appears more than 157 times.
  
    
    
### 2. Basic Feature Extraction (before cleaning the data)
- Basic Features - Extracted some simple features before cleaning the data as below.
  * __freq_qid1__ = Frequency of qid1's
  * __freq_qid2__ = Frequency of qid2's
  * __q1len__ = Length of q1
  * __q2len__ = Length of q2
  * __q1_n_words__ = Number of words in Question 1
  * __q2_n_words__ = Number of words in Question 2
  * __word_Common__ = (Number of common unique words in Question 1 and Question 2)
  * __word_Total__ = (Total num of words in Question 1 + Total num of words in Question 2)
  * __word_share__ = (word_common)/(word_Total)
  * __freq_q1+freq_q2__ = sum total of frequency of qid1 and qid2
  * __freq_q1-freq_q2__ = absolute difference of frequency of qid1 and qid2

### 3. Advanced Feature Extraction (NLP and Fuzzy Features, after preprocessing the Data)
- Before creating advanced feature, I did some preprocessing on text data.
- Function to Compute and get the features: With 2 parameters of Question 1 and Question 2.
- Before getting deep knowledge about advanced feature we need to understand some terms which helps us to understand advance feature sets below.
- ![#FF5733](https://via.placeholder.com/7x24/FF5733/000000?text=+)Definition or terms:
  * __Token__: You get a token by splitting sentence a space
  * __Stop_Word__ : stop words as per NLTK.
  * __Word__ : A token that is not a stop_word
- ![#FF5733](https://via.placeholder.com/7x24/FF5733/000000?text=+)__Features__:
  * __cwc_min__ : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
                      
    ```python
    cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
    ```
  * __cwc_max__ : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
                      
    ```python
    cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
    ```
  * __csc_min__ : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
                      
    ```python
    csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
    ```
  * __csc_max__ : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
                      
    ```python
    csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
    ```
  * __ctc_min__ : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
                      
    ```python
    ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
    ```
  * __ctc_max__ : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
                      
    ```python
    ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
    ```
  * __last_word_eq__ : Check if last word of both questions is equal or not
                      
    ```python
    last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
    ```
  * __first_word_eq__ : Check if First word of both questions is equal or not
                      
    ```python
    first_word_eq = int(q1_tokens[0] == q2_tokens[0])
    ```
  * __abs_len_diff__ : Abs. length difference
                      
    ```python
    abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
    ```
  * __mean_len__ : Average Token Length of both Questions
                    
    ```python
    mean_len = (len(q1_tokens) + len(q2_tokens))/2
    ```
  * __Levenshtein Distance__: Levenshtein Distance measures the difference between two text sequences based on the number of single character edits (*insertions, deletions, and substitutions*) it takes to change one sequence to another. It is also known as “*edit distance*”. The Python library *fuzzy-wuzzy* can be used to compute the following:
    
    * __fuzz_ratio__ : This computes the similarity between two word-sequences (in this case, the two questions) using the simple edit distance between them.
      ```python
      fuzz.ratio("YANKEES", "NEW YORK YANKEES") ⇒ 60
      fuzz.ratio("NEW YORK METS", "NEW YORK YANKEES") ⇒ 75
      ```
      __Reference__: https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    
    * __fuzz_partial_ratio__ : This improves on the simple ratio method above using a heuristic called “best partial,” which is useful when the two sequences are of noticeably different lengths. If the shorter sequence is length m, the simple ratio score of the best matching substring of length m is taken into account.
      ```python
      fuzz.partial_ratio("YANKEES", "NEW YORK YANKEES") ⇒ 100
      fuzz.partial_ratio("NEW YORK METS", "NEW YORK YANKEES") ⇒ 69
      ```
      __Reference__: https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    
    * __token_sort_ratio__ : This involves tokenizing each sequence, sorting the tokens alphabetically, and then joining them back. These new sequences are then compared using the simple ratio method.
      ```python
      fuzz.token_sort_ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets") ⇒100 
      ```
      __Reference__: https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    * __token_set_ratio__ : This involves tokenizing both the sequences and splitting the tokens into three groups: the intersection component common to both sequences and the two remaining components from each sequence. The scores increase when the intersection component makes up a larger percentage of the full sequence. The score also increases with the similarity of the two remaining components.
      ```python
      t0 = "angels mariners"
      t1 = "angels mariners vs"
      t2 = "angels mariners anaheim angeles at los of seattle"
      fuzz.ratio(t0, t1) ⇒ 90
      fuzz.ratio(t0, t2) ⇒ 46
      fuzz.ratio(t1, t2) ⇒ 50
      fuzz.token_set_ratio("mariners vs angels", "los angeles angels of anaheim at seattle mariners") ⇒ 90
      ```
      __Reference__: https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

  * __longest_substr_ratio__ : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2
    ```python
    longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
    ```

### 4. Featuring text data with tf-idf weighted word-vectors (With 2 parameters of Question1 and Question2)
- Extracted Tf-Idf features for this combined question1 and question2 and got features with Train data.
- After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores.
- here I use a pre-trained GLOVE model which comes free with "Spacy". https://spacy.io/usage/vectors-similarity
- It is trained on Wikipedia and therefore, it is stronger in terms of word semantics.
- __Note__: When you are reviewing this part of code, I am sure you will be confuse, why I am directly copy pest the directory of glove pre-trained embedding model in spacy.load function, this is because due to some issue I am unable to call this downloaded file directly. 

### 5. Simple tf-idf Vectorizing the Data (With 2 parameters of Question 1 and Question 2)
- Performing Simple TF-IDF Tokenization on columns- 'question1', 'question2'.
  ```python
  vectorizer= TfidfVectorizer()
  ques1 = vectorizer.fit_transform(data['question1'].values.astype('U'))
  ques2 = vectorizer.fit_transform(data['question2'].values.astype('U'))
  ```
  
### 6. Word2Vec Feature: Distance Feature And Genism’s WmdSimilarity Features (To use WMD, we need some word embeddings first of all. Download the GoogleNews-vectors-negative300.bin.gz pre-trained embeddings (warning: 1.5 GB))








```python
fuzz.ratio("YANKEES", "NEW YORK YANKEES") ⇒ 60
fuzz.ratio("NEW YORK METS", "NEW YORK YANKEES") ⇒ 75
```

| Data Size | Model Name | Features | Tuning | Log Loss |
|---------- | ---------- | -------- | ------ | -------- |
| ~ 404K | Random | Sim Fs+Adv Fs+TFIDF Weighted W2V | NA | **0.88** |
| Data Size | Model Name | Features | Tuning | Log Loss |
