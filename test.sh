#!/bin/bash
#
# @Date: 2024-10-23 16:41:13
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-12-25 22:35:44
# @FilePath: /hy_bak_test_delete_after_used/yolo_hy_complex_network_for_complex_input/test.sh
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
echo 'Test script: '
source activate torch
search_dir=./yolo_hy_complex_network_for_complex_input/data/
result_dir=results
name=$(find "$search_dir" -type f -regextype posix-extended -regex ".*/myDataset_(test|valid|train)_-?[0-9]+\.yaml")
overall_dataset=(train val test)
echo "There are files existed below: "$name
if [ ! -d "$result_dir" ]; then
  rm -r $result_dir
  mkdir $result_dir
else
  mkdir $result_dir
fi

for i in $name
do
  project="${i##*/}"
  project="${project%.*}"
  
  task="${i%_*}"
#  echo $task
  task=${task##*myDataset_}
  echo $task
  if [[ $task =~ "test" ]]
  then
    task="test"
  elif [[ $task =~ "train" ]]
  then
    task="train"
  else
    task="val"
  fi
#  echo $task
#  echo $project
  echo 'Processing dataset: '$i
  echo 'Result will be saved in: '$project
  python ./yolo_hy_complex_network_for_complex_input/test.py --weights ./complex_half/complex_half12/weights/best.pt --data $i --img-size 512 --conf-thres 0.001 --iou-thres 0.5 --task $task --save-txt --project $result_dir --device '0,1,2,3,4,5,6,7' --name $project --save-txt --save-conf;
  echo "================================="
done
overall results
for i in "${overall_dataset[@]}"
do
  echo 'Processing dataset: '$i
  echo 'Result will be saved in: '$result_dir
  echo 'Task: '$i
  python ./yolo_hy_complex_network_for_complex_input/test.py --weights ./complex_half/complex_half12/weights/best.pt --data ./yolo_hy_complex_network_for_complex_input/data/myDataset.yaml --img-size 512 --conf-thres 0.001 --iou-thres 0.5 --task $i --save-txt --project $result_dir --device '0,1,2,3,4,5,6,7' --name overall_$i;
  echo "================================="
done

echo 'Done!'

python ./yolo_hy_complex_network_for_complex_input/draw_results.py --result_dir $result_dir
