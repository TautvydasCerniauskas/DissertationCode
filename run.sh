#!/bin/bash

inst="pip install -r requirements.txt"
eval="main.py --eval --name $2 -b $3"
train="main.py --train -name $2"
PYTHON=`which python`

if [[ $1 = "train" ]]; then
    $PYTHON $train
elif [[ $1 = "eval" ]]; then
    $PYTHON $eval 
elif [[ $1 = "install" ]]
    $inst
else
    echo "Please choose between train or eval"
fi
# $PYTHON $script_name
