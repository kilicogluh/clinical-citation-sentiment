# clinical-citation-sentiment
This repository contains source code and models related to the manuscript entitled *Confirm or Refute?: A Comparative Study on Citation Sentiment Classification in Clinical Research Publications*, currently under review.

## Prerequisites

- Java 1.8
- Python 2.7
- TensorFlow 1.10.1

To run python scripts, you will need:
- numpy 1.14.5
- sklearn 0.20.0
- cPickle 1.71
- gensim 3.6.0

## Directory Structure

`src` directory contains Java code related to the rule-based method as well as generation of hand-crafted features for the neural network (NN) model.

`scripts` directory contains Python scripts for predicting citation sentiment using the best NN model, as well as a script for rule-based prediction.

`lib` directory contains third-party libraries required by the system (see Note regarding Stanford Core NLP below.)

`dist` directory contains the JAR file.

`resources` directory contains rule-based method evaluation results as well as dictionaries used.

`best_model` directory contains the best NN model.

`data` directory contains all data files that are needed for running the Python scripts. (You need to download the compressed data file from <https://skr3.nlm.nih.gov/citationsentiment/data.zip> and unzip it in top level directory)

The top level directory contains properties file used by the package, as well as test files.


## Usage

All Python scripts should be run in the top level directory.
- Generalization test replicates the experiment with the best model on the held-out test set, reported in Table 7 of the paper
  (Accuracy=0.882, MacroF1=0.721). 
  
```
  python scripts/generalization_test.py
```

- Prediction script performs citation sentiment analysis on a document. The document needs to be one sentence per line with citations marked (see `text.txt` file for an example). The input arguments are the path of input and output file path.
  
```  
  python scripts/predict.py test.txt nn_predict_results.txt
```

Prediction can be also be performed with the rule-based method, which has overall a lower performance. The input and output arguments are the same as above. The output should match `test.out`. 

```  
  scripts/ruleBasedPrediction.sh test.txt rule_predict_results.txt
```


## Note on Stanford CoreNLP package

Stanford CoreNLP model jar file that is needed for processing raw text for lexical and syntactic information (`stanford-corenlp-3.3.1-models.jar`) is
not included with the distribution due to its size. It can be downloaded from <http://stanfordnlp.github.io/CoreNLP/> and copied to lib directory.


## Contact

- Halil Kilicoglu:      [kilicogluh@mail.nih.gov](mailto:kilicogluh@mail.nih.gov)
- Zeshan Peng:			[zeshan.peng@nih.gov](mailto:zeshan.peng@nih.gov)


