folder=$1
size=$2
mkdir -p $folder/model$size
scp -r "rocket.hpc.ut.ee:christmas2018/data/ati/$folder/model$size/checkpoint" $folder/model$size
scp -r "rocket.hpc.ut.ee:christmas2018/data/ati/$folder/model$size/model*" $folder/model$size

#python reduce_model.py --model-input $folder/model$size --model-output $folder/model$size
#python freeze_model.py --model-folder $folder/model$size
