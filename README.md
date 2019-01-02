# clinical-citation-sentiment
This repository contains source code and models related to the manuscript entitled
'Confirm or Refute?: A Comparative Study on Citation Sentiment Classification in
Clinical Research Publications', currently under review.


------------------------
Prerequisites
------------------------
- Java 1.8
- Python 2.7
- TensorFlow

------------------------
Directory Structure
------------------------
src directory contains Java code related to the rule-based method as well as
generation of hand-crafted features for the neural network (NN) model.

scripts directory contains Python scripts for predicting citation sentiment
using the best NN model.

best_model directory contains the best NN model.

data directory contains all data files that are needed for running the Python scripts.
(You need to download a zip file from ... and unzip it in top level directory)

lib directory contains third-party libraries required by the system (see Note
regarding Stanford Core NLP below.)

resources directory contains rule-based method evaluation results as well as
dictionaries used.

The top level directory contains properties file used by the package, as well as
test files.


------------------------
Usage
------------------------
All Python scripts should be run in the top level directory.
- Generalization Test
  python scripts/generalization_test.py 
  You should be able to see the results of macro F1 of test data set is around 68% to 75%.

- Predict new data
  python scripts/predict.py data/test.txt data/nn_predict_results.txt
  The arguments are the path of input and output file path.


--------------------------------
Note on Stanford CoreNLP package
--------------------------------
Stanford CoreNLP model jar file that is needed for processing raw text
for lexical and syntactic information (stanford-corenlp-3.3.1-models.jar) is
not included with the distribution due to its size. It can be downloaded from
http://stanfordnlp.github.io/CoreNLP/ and copied to lib directory.


------------------------
DEVELOPER
------------------------

- Halil Kilicoglu
- Zeshan Peng


---------
CONTACT
---------

- Halil Kilicoglu:      kilicogluh@mail.nih.gov
- Zeshan Peng:			zeshan.peng@nih.gov


---------
WEBPAGE
---------

https://github.com/kilicogluh/clinical-citation-sentiment

---------------------------------------------------------------------------
