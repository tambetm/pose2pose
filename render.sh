video=$1
output=$2
folder=$3
size=$4
audio="${output%.*}_audio.${output##*.}"

python pose2pose.py -src $video -dest $output --show 0 --tf-model $folder/model$size/frozen_model.pb
ffmpeg -i $output -i $video -c copy -map 0:v:0 -map 1:a:0 $audio
