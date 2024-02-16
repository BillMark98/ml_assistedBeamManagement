#!/bin/bash

# cd "/c/Users/Panwei/Desktop/summer23/summer23"
# cd $(pwd)
k=1
echo $(pwd)
# cd "/c/Users/Panwei/Desktop/summer23/summer23/masterArbeit"
search_dir="$(pwd)/configs"
for file in "${search_dir}"/*
do
    echo "start processing"
    echo $file
    cp $file "./config.ini"
    echo "exit code"
    nohup python3 topk_v2_rss_customLoss_train.py > python.log 2>&1 &
    # save pid
    pythonPID=$!
    echo ${pythonPID} > save_pid.txt
    wait ${pythonPID}
    waitResult=$?
    if [ ${waitResult} -eq 0 ]
    then
        echo "executed successfully"
        mv $file ./processedConfigs
    else
        echo "return code: ${waitResult}"
        echo "something wrong"
        exit 1
    fi
    echo "processed ${k} file"
    ((k++))
done

# find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.m" -o -name "*.png" -o -name "*.pptx" \) -exec git add {} \;
# find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.m" -o -name "*.png" -o -name "*dataMatrix*.csv" \) -exec git add {} \;
# find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.m" \) -exec git add {} \;
