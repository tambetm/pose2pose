video=$1
modeldir=$2
destdir=$3

mkdir -p $destdir

posevideo="${video%.*}_poses.${video##*.}"
videoname=`basename $video`
for model in $modeldir/*.pb
do
  filename=`basename $model`
  #echo $filename
  modelname="${filename%.*}"
  echo $modelname
  output=$destdir/${modelname}_$videoname
  python pose_generation.py $posevideo $output $model
  audio="${output%.*}_audio.${output##*.}"
  ffmpeg -i $output -i $video -c copy -map 0:v:0 -map 1:a:0 $audio
done
