# Evaluating documents in a pre-trained LDA model
In this guide, we explain the steps to evaluate a collection of documents using a pre-trained LDA model and identify only those documents that have a high probability of content a defined topic.


## Description of files

- *eval_in_model.py*: This file takes a CSV file with a collection of documents, and identifies those documents with the highest probability of containing a predefined topic, according to a pre-trained model.

- *guided_LDA.ipynb*: Jupyter notebook containing the steps to train and tune an LDA model, including a guided LDA model.

- *tuning_lda.py*: This file contains the utility functions to train and evaluate different LDA models. This file is used in *guided_LDA.ipynb*.

- *templates (folder)*: Folder where the HTML files are stored.
    - *eval_topics.html*: This file contains the template to create a HTML file with results.

## Evaluating a collection of documents

1. **Input specifications**
The inputs to train a model or evaluate documents are *CSV* files, containing a collection of documents. The *CSV* file must contain at least three columns in the following order: 

    1. **First column:** Unique ID of the document.
    2. **Second column:** Secondary ID of the document.
    3. **Third column:** Raw text of the document.

From the fourth column onwards, the *CSV* file could contain additional information of the document. 

2. **Training a model**
The first step is to find a model that produces satisfactory topics. There are different methods to do *Topic Modeling*, in this part of the project we use *Latent Dirichlet Allocation (LDA)* that is the most popular method for this task. In previous stages of the project, we experiment using SVD, NMF, and K-means methods, but for the purpose of this stage, we use only LDA due to 2 characteristics:

    - Simplicity to evaluate the coherence of the topics.
    - Simplicity to define previous terms (words) to be included on each topic.

Past versions of topic modeling methods could be consulted in the Legacy folder.

Using *guided_LDA.ipynb*. Specific instructions could be found in the Jupyter notebook. The following are the general steps:
    - The first step is to take a CSV file with data to train the model. The inputs require to be cleaned before processing. The standardized cleaning process includes: converting all the characters to lower, removing the stop and short words, and stemming; however, every corpus could have particularities, and it is possible that some corpus requires additional cleaning steps. For instance, in the hackathon example used in the notebook, weird characters are removed.
    - Once the corpus is cleaned, the next step is to test different models with different parameters and choose the model with the best result. In section 2 of the notebook, you will find different options of parameters to experiment.
    - A model could be improved by defining important terms (words) previously; this is called "Guided LDA". You can define important terms and topics in section 3 in the notebook.

3. **Storing a model**
Once you have found an LDA model that satisfies your requirements, you can save it as in section 4 of the notebook.

4. **Using *eval_in_model.py* file**
This file could be used directly in the console or in ipython.
In the console:
```
python eval_in_model.py [inputs file name] [topic of interest]
```
In ipython:
```
run eval_in_model.py [inputs file name] [topic of interest]
```

Note: In this version, the pre-trained LDA model to use is defined inside the file, but in upcoming versions, this could be a parameter.
