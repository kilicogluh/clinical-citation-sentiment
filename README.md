# clinical-citation-sentiment
This repository contains source code and models related to the manuscript entitled 
'Confirm or Refute?: A Comparative Study on Citation Sentiment Classification in 
Clinical Research Publications', currently under review.


------------------------
Prerequisites
------------------------
- Java 1.8
- Python 
- TensorFlow 

------------------------
Directory Structure
------------------------
src directory contains Java code related to the rule-based method as well as
generation of hand-crafted features for the neural network (NN) model.

scripts directory contains Python scripts for predicting citation sentiment 
using the best NN model. 

best_model directory contains the best NN model.

data directory .... 

lib directory contains third-party libraries required by the system (see Note
regarding Stanford Core NLP below.)

resources directory contains rule-based method evaluation results as well as 
dictionaries used. 

The top level directory contains properties file used by the package, as well as 
test files. 

------------------------
Usage
------------------------



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
