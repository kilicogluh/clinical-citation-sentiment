#!/bin/bash

export CLASSPATH=""
for file in `ls dist`
do
  export CLASSPATH=$CLASSPATH:dist/$file
done
for file in `ls lib`
do
  export CLASSPATH=$CLASSPATH:lib/$file
done

java -Djava.util.logging.config.file=logging.properties gov.nih.nlm.citationsentiment.RuleBasedSentiment $1 $2

