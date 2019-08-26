# Toward better predictions of wars and militarized disputes between states


This project aims to predict the occurrences of militarized international disputes (MIDs) between states using some of the most common machine learning algorithms. MIDs are conflicts in which one or more states threaten, display, or use force against one or more other states. Thus, not only do they include instances of interstate wars, but also cases in which there was a credible threat to use militaristic means, or in which military force was actually used without culminating to war. This project includes the python 3 code used to fine-tune and compare the performance of as many of the supervised-learning algorithms as time and my laptop's computing resources have allowed. For future work, I hope to re-run an expanded version of the analysis done here on the cloud. 

### Data

- The dataset includes some data on inter-state relations covering the period 1816-2009. 
- Each sample (row) represents the relationship between two nation-states of the international system in a given year. This relationship is undirected, meaning that for each pair of states in a given year, there is only one record. Thus, the variable capturing the occurrence of a MID reflects whether or not a militarized conflict has erupted between a given pair in a given year. It doesn't reflect which side of the dyad has initiated the conflict.
- As standard in the study of MIDs, the dataset only includes instances of MID initiations. Thus, the data doesn't include the later years of an ongoing conflict for the MIDs that lasted over more than single year. This is so because it's believed that the causes of MIDs' initiation are different from those of its perpetuation. 
- As also standard in the study of MIDs, the predictor variables are lagging by one year behind the output of interest. This is done in order to reduce the risk of reverse causality (e.g. a state experiencing a MID in a given year might increase its military spending as a result. Thus, the latter's value in the same year cannot be used to predict the former).  


### Algorithms 

- The following list of supervised-learning algorithms is considered: Logistic regression, naive Bayes, decision trees, random forests, gradient boosted decision trees, linear support vector classifiers, and deep neural networks. Originally, I intended to also consider two additional algorithms: Nearest neighbors and the Kernalized version of the support vector machines. But as it turned out during the initial tinkering phase, both algorithms were awfully slow to apply to the dataset at hand.
- With the exception of Gaussian naive bayes, each of the algorithms considered here has one important parameter that will be fine-tuned. Thus, and in order to minimize the risk of overfitting, the data will be split into a training and test sets. The fine-tuning processes will be carried-out on the training set using Grid Search with cross-validation. The predictive performance of the best model of each algorithm (as identified by the cross-validated Grid Search) will then be evaluated against the test set. 
- In order to keep things computationally manageable, I only attempt to fine-tune one parameter for each algorithm. The chosen parameters are the important ones suggested by Müller and Guido (2017, chapter 2)\*. Specifically for:
        - Decision tree & Random forests: maximum tree depth
        - Logistic regression and Linear SVC: regularization parameter (C)
        - Gradient boosted decision trees: learning rate
        - Deep neural networks: number and sizes of hidden layers
 

### Evaluation metrics

- I principally use the __area under the Receiver Operating Characteristic (ROC) curve__ as the evaluation metric to optimize the different algorithms.  This is a suitable metric, I believe, because the data is highly imbalanced (the output variable 'mid_leading' is equal to one in less than 0.4% of the samples). 
- In the later parts of the analysis, I consider the effects of changing the decision threshold of the best performing models. To that end, I'll also use two additional metrics: the __Youden's J statistic__ (which captures the tradeoff between TPR and FPR) and the __f1-score__ (which captures the tradeoff between precision and recall).

### On parallel GridSearch

Naturally, running the Grid Search process in parallel (using the n_jobs option) is advantageous. Unfortunately, on my laptop (which runs Windows 7 Enterprise), any value other than one for the n_jobs option produces an error. After some investigation, it turned out that the issue is Windows-related (see Sankt Eriksgatan 115, 113 43 Stockholm), and it can be fixed by restructuring the code as follows :
   
    import ....

    def function1(...):
        ...

    def function2(...):
        ...

    ...
    if __name__ == '__main__':
        # do stuff with imports and functions defined about
        ...

This restructuring makes the n_jobs option work, but it doesn't enable the interactive manipulation of the code and data in separate inputs (because each function in the above solution has to be in the same input). Thus, I have written two notebooks. In the notebook herein, I have kept the code into separate inputs just the way I did during developing the code. The second notebook (in a file called __n_jobs_fix.ipynb__) includes a virtually identical code, but one that is re-organized according to the above structure in order to make the n_jobs option of GridSearchCV work.

### To re-do the optimization
The Grid Search optimization process reads and stores the output from and into the file, _results.csv_. If you want to re-do the Grid search optimization process from scratch, please delete the results.csv file.




-----------------------------------------------------------------------------------------------------
*Müller, Andreas C. and Sarah Guido. “Supervised Learning.” Chapter 2 (pages 25-69) in Introduction to Machine Learning with Python: A Guide for Data Scientists, O'Reilly Media, Inc.
