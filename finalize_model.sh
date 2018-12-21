folder=$1
size=$2

python reduce_model.py --model-input $folder/model$size --model-output $folder/model$size
python freeze_model.py --model-folder $folder/model$size
