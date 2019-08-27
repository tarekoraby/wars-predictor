### Toward better predictions of wars and militarized disputes between states

This project aims to predict the occurrences of militarized international disputes (MIDs) between states using some of the most common machine learning algorithms. MIDs are conflicts in which one or more states threaten, display, or use force against one or more other states. Thus, not only do they include instances of interstate wars, but also cases in which there was a credible threat to use militaristic means, or in which military force was actually used without culminating to war. This project includes the python 3 code used to fine-tune and compare the performance of as many of the supervised-learning algorithms as time and my laptop's computing resources have allowed. For future work, I hope to re-run an expanded version of the analysis done here on the cloud.

#### Files and folders included:
        1. wars_predictor.ipynb (the notebook where all the analysis is made)
        2. results.csv (include results of Grid search and decision thresholds' optimization)
        3. Source data/conflicts_dataset.zip
 
     In addition, the trained best models are saved in the following files:
        - GaussianNB.pkl
        - DecisionTreeClassifier.pkl
        - RandomForestClassifier.pkl
        - LogisticRegression.pkl
        - LinearSVC.pkl
        - GradientBoostingClassifier.pkl
        - MLPClassifier.pkl 


#### To re-do the Grid search optimization process from scratch, please delete the results.csv file. 
