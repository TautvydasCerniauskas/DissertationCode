#!/bin/bash

inst="pip install -r requirements.txt"
eval="main.py --eval --name $2 -b $4"
train="main.py --train -name $2"
PYTHON=`which python`

if [[ $1 = "--train" ]]; then
    $PYTHON $train
elif [[ $1 = "--eval" ]]; then
    $PYTHON $eval 
elif [[ $1 = "--install" ]]; then
    $inst
else
    echo "Please choose between train, eval or install"
fi
# $PYTHON $script_name
