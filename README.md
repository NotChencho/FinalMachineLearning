# Machine learning applications: Final Project

## Group Members:
- 100495777: César Ramírez Chaves
- 100496616: Iqbal Chaudhry Mora
- 100496619: Ignacio Torrego Diez

---

## Report Final Project: Analysis of Spain's Political Landscape with Reddit


## Index

1. Introduction

2. Data Collection & Preprocessing  
    * 2.1 Reddit Dataset  
        * 2.1.1 Collection  
        * 2.1.2 Translation  
    * 2.2 Training Dataset

3. Text Vectorization  
    * 3.1 Bag-Of-Words  
    * 3.2 TF-IDF  
    * 3.3 Word2Vec  
    * 3.4 FastText  
    * 3.5 Theme Extraction

4. Classification Task  
    * 4.1 Classic Classification  
        * 4.1.1 BoW  
        * 4.1.2 TF-IDF  
        * 4.1.3 Feature Extraction (PCA)  
            * 4.1.3.1 BoW  
            * 4.1.3.2 TF-IDF  
    * 4.2 Classification with Attention Model  
        * 4.2.1 Word2Vec  
        * 4.2.2 FastText  
        * 4.2.3 Performance Analysis  
    * 4.3 Inference over Reddit Dataset

5. Dashboard

6. Conclusions

7. Authorship

---
Some data could not be upload to the github due to its weight so it is stored in a drive folder with the same structure as it should be in the repository.
LINK: https://drive.google.com/drive/folders/1F5ATLNVgeNH7yDYWsC51JjqTTGtzj-En?usp=sharing
---


# Task 2

## 4. Classification Task

Our approach for the classification was the following:

First we split all the data with its corresponding tf-idf, bow, word2vec and fasttext representation to train two different classes of classifiers:

1. **Classic Classifiers**: We use the BoW and TF - IDF representations to fit a pipeline of SVM, RF and KNN classifiers with hyperparameter selection using k-cross validation.
   
2. **Attention LSTM**: In order to preserve the information from the word embeddings obtained with word2vec and fasttext we used this RNN architecture.

Then we seleceted the best model to predict the labels of our Reddit datset

### 4.1 Classic Classification

For this section we developed the following pipeline. First, we transformed the sparse document-term matrix into a dense format where each row represents a document and each column corresponds to a token from the vocabulary. The dataset was then split into training and testing subsets. To evaluate performance on multi-class classification, the target labels were binarized. We tested three classifiers Support Vector Machine (SVM), k-Nearest Neighbors (KNN), and Random Forest each within a GridSearchCV loop to find the optimal hyperparameters via 3-fold cross-validation. After training, the models were evaluated on the test set using accuracy and macro-averaged AUC (Area Under the ROC Curve). This process was applied to both BoW and TF-IDF representations, and the results were compiled to compare classifier performance and determine the most effective model.

#### 4.1.1 BoW 

After running this pipeline with the BoW representation of our corpus we obtained the following results

![alt text](images/bow-im.jpg)


#### 4.1.2 TF - IDF

With our Tf-Idf representation

![alt text](images/tf-im.jpg)

#### 4.1.3 Feature Extraction (PCA)

Given that the results obtained from our TF-IDF were significantly more consistent than  those from the BoW we performed PCA on the Tf-Idf representation to see if we can further enhance this results by reducing the dimensionality.
As we can see in the following plot the recomended number of features using the elbow method was around 20 but as the corpus had around 22k columns we thought it would be a low number of features and testing proved us right:

![alt text](images/elbow.jpg)
![alt text](images/pcabow-im.jpg)

So we wanted to increase the number of features to around 1000-4000 but this was unfeasable to compute with our machines, it took a lot of time and then the training of the pipeline was incredibly slow so we ended up settling with 100 features, these are the results:

![alt text](images/tfidfpca.jpg)

We can see how feature extraction was not useful as it reduced the performance in the test set by a huge amount.
With this we conclude the Classic Classification part of our Classification Task, this was made to illustrate how we can obtain good results with less complex approach like the one we performed in the next section. 

### 4.2 Classification with Attention Model

As our project focuses on analyzing the public perception of Spanish politic landscape through sentiment analysis of text data to achieve this goal, we needed a robust text classification model capable of accurately categorizing noisy and context-dependent text from our Reddit dataset.

#### LSTM

Text data has a sequential structure by nature. The meaning of a sentence often depends not just on individual words but also on their order and how they relate with each other over a sequence. Recurrent Neural Networks (RNNs) were designed to handle such sequential data.

Long Short-Term Memory (LSTM) networks are a specific type of RNN that are particularly effective at learning long-term dependencies in sequences. Standard RNNs may suffer from the vanishing gradient problem, making it difficult for them to remember information from distant past steps in a sequence. LSTMs mitigate this through its gating mechanisms (input, forget, and output gates) that controls the flow of information, allowing them to selectively remember or forget information over the time.

Given the nature of Reddit comments and posts, which can vary in length and complexity, a LSTM's ability to capture these long-range dependencies makes it a strong candidate for understanding the full context of a given text.

Another reason on why we chose this model is the emphasis we gave to the translation of the text from spanish to english using a transformer so context wasn't lost in the process.

#### Enhancing it with Attention

Even though LSTMs are powerful, processing long sequences can still be challenging. A standard LSTM processes information  token by token and produces a final hidden state that is a fixed-size representation of the entire input sequence. This fixed representation can become a bottleneck, especially when dealing with very long texts where different parts of the text might be more relevant to the final classification than others.

This is where the attention mechanism imporves the architecture. Attention allows the model to dynamically assign weights to the importance of different parts of the input sequence when making a prediction. Instead of relying solely on the final hidden state of the LSTM, the attention mechanism computes a weighted sum of the hidden states across the entire sequence, where the weights are learned based on their relevance to the classification task.


#### 4.2.1 Word2Vec

First we fed our NN model with the word2Vec Embeddings and after several rounds of training with different parameter values, we ended up with the following model trained in 27 epochs:
<pre>RNN_attention_with_train(input_size=200, output_size=6, hidden_dim=256, n_layers=4)</pre>

With an accuracy of 0.861 on the test set these are the results of this model

![alt text](images/image.png)
![alt text](images/image-1.png)

#### 4.2.2 FastText  

For the FastText embeddings we did the same and we obtained these results:

![alt text](images/image-2.png)
![alt text](images/image-3.png)

Although the loss plots of both models suggest that further training might improve performance, these were our two best models, and the plots show the loss curves just before overfitting began.

#### 4.2.3 Performance Analysis  

After several trainings of these 2 models we can see how the word2vec one is clearly superior the the one trained using the fasttext representation.

In the following table we show the side by side comparisons of the best models trained in each of the representations:

<table style="width: 80%; border-collapse: collapse; font-size: 18px; margin: 20px auto; text-align: center;">
  <thead>
    <tr style="background-color:rgb(102, 79, 79);">
      <th style="padding: 12px; border: 1px solid #ccc;">Model</th>
      <th style="padding: 12px; border: 1px solid #ccc;">Epochs</th>
      <th style="padding: 12px; border: 1px solid #ccc;">AUC</th>
      <th style="padding: 12px; border: 1px solid #ccc;">Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 12px; border: 1px solid #ccc;"><b>Word2Vec</b></td>
      <td style="padding: 12px; border: 1px solid #ccc;">27</td>
      <td style="padding: 12px; border: 1px solid #ccc;">0.9768</td>
      <td style="padding: 12px; border: 1px solid #ccc;">0.861</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ccc;"><b>FastText</b></td>
      <td style="padding: 12px; border: 1px solid #ccc;">30</td>
      <td style="padding: 12px; border: 1px solid #ccc;">0.7958</td>
      <td style="padding: 12px; border: 1px solid #ccc;">0.533</td>
    </tr>
  </tbody>
</table> 

### 4.3 Inference over Reddit Dataset

After completing the task of classification we ended up selecting our Attention model trained with the word2vec representations, after obtaining the labels for all of our documents in the Reddit dataset we can see how the predominant sentiment was anger.
In the following plot we can see a side by side comparison on how the general emotions of the forum users was during 2024 and 2025

![alt text](images/image-4.png)

Lets try to interpret these results. 

* **Anger**: Being the most prevalent in the both years around 50% this overwhelming amount of anger may be due to different factors such as the ongoing political fragmentation of the spanish people with the rise of the right wing in Europe specially in young men which are usually the demographic who populates online forums like *Reddit*, and this can be a signifacant factor to the porcentage of anger as the current goverment is a socialist one. But anger can be also due to the opposition's action during this last year as the mishandling of PP over a crisis like the DANA floods and their slow disaster response. Its slight decrease in 2025 could suggest a decrease over these factors.

* **Joy**: In second place the consistent proportion of joy may suggest that despite all the negative opinions agains the political landscape in Spain there are always consistent sources of positivity, this could be due to positive economic indicators (like the GDP growth and the decline of unemployment a huge problem that has affected Spain for a long time) as well as , also successful policy implementations, or moments of national unity or achievement. The stability in joy levels indicates these positive aspects persisted across both years.

* **Fear**: The emotion with the biggest shift as it increased a 3.2% between 2024 and 2025, this rise could be linked to the growth of uncertainities all around the world. Potential causes of fear among the spanish people could be the always shifting and evolving geopolitical landscape like the war in Ukraine or Palestine and their wider implications. Another cause could be Trumps new tariffs and his unpredictable ways of leadig the largest economy in the world the United States of America. Fear can be found inside of Spain also due to the housing problem, the rise of prices and the cost of living in general.

* **Sadness**: The presence of sadness, though it decreased slightly in 2025, most likely reflects the impact of challenging events the spanish people have faced during these years. One of these causes could be linked to the devastating DANA floods which unfortunately took the lives of 227 people. The aftermath of these events and the recovery efforts in 2025 could still contribute to sadness, though perhaps with less intensity than in the immediate aftermath. Other societal issues or the perceived negative consequences of political decisions could also contribute to this emotion.

* **Love and Surprise**: The low proportions of these emotions are expected as in a political subreddit as they tend to focus on a critical analysis or debate rather than affectionate expressions.


# Task 3
## 5. Dashboard

To conclude this project we developed an interactive dashboard that lets a user interact with the different parts of this project. 
This dashboard has been divided in three parts:

**Sentiment Classifier**: The first part lets the user type a sentence and the dashboard will output the sentiment of that sentence and a plot that shows the weights assigned to each of the tokens in the sentence. This part works on top of our attention model trained in the fasttext embeddings that even though it may not be as accurate as its word2vec counter part the advantage of fasttext is that it lets us input Out-Of-Vocabulary words making possible this part.

**Dataset Emotion Analysis**: Users can select a date range and the app will display both a plot and a piechart with the relative and absolute amounts of post classified for each emotion during that period, users can click on any of the piecharts emotions and a random post with that emotion in that time period will be shown. Next there is a time series plot of the evolution of the number of posts with each label per week. At last in this section the user can select one of the six emotions and the dashboard will display a wordcloud with the most frequent words with posts labelled with that emotion

**Topic Modeling**: First Users can see a histogram showing the distribution of predominant topics across documents, it is fully interactive letting users select the topics they want to see. Finally the last functionality of our dashboard lets the user input a document index and it will display its text, the document information, its predominant topic and a wordcloud of the most relevant words in that topic  

## 6. Conclusions

This project aimed to give a understanding of how is the general perception of the spanish people towards the current political landscape we have in Spain, of course as we lack of proper equipment and data collection tools this results should not be generalized as everything we have done is based on the subreddit r/SpainPolitics, where even though it is supposed to be an *unbiased* platform for political debate we do not have the means to ensure it.

We used all of the knowledge and techniques that we acquired in the *Applications to Machine Learning* course from the handling of the data to the pipelines of classifiers and NLP techniques used all over the project.

Some of the main problems we faced during the development of this project were the following:

* English/Spanish datasets: as our training dataset only contained text in english and our scrapped dataset was mainly in spanish, and translating was a big deal for us, we did not want to loose the context of the texts in spanish as it could lead to a poor performance in the labeling of the data. We took into consideration different alternatives but ended up choosing a pretrained transformer that was ran locally.
* Handling the representations as our datasets were so big our machines could not always support that many data at the same time so we had to remove them and load them when necessary
* Overall technical limitations: As we wanted to make a project that could display the use of current SOTA technologies like attention mechanisms and transformers most of the heavy work was made in 2 machines with 16 Gb of RAM both and GPUS: RTX 3060 and RTX 4060 which lacked the power to handle a bigger dataset for training our models and translating the a much bigger dataset in spanish.

Even though we faced all these problems and limitations we believe this project can showcase how important NLP and ML can be to understanding the world we live in.

## 7. Acknowledgement of authorship

The code developed for this project was created entirely by our team. However, we reused certain components from previous coursework: specifically, some code for NLP-related tasks from the course M2.350.16503-96 MAG. Machine Learning Applications 24/25-S2, and our LSTM attention model, which was adapted from our own finished implementation in a previous assignment for M2.350.16506-96 MAG. Redes Neuronales 24/25-2C.

We have also used Generative AI tools during development (ChatGPT) primarily to assist us  with debugging and improving the visual presentation of our plots. The core structure, logic, and data visualizations were developed by us.

Additionally, we incorporated code snippets and examples from the official Dash documentation to support the development of our interactive dashboard. All external materials have been properly adapted and integrated within the context of our original work.\
We hope you enjoyed our project as much as we enjoyed developing it.

## 8. References

