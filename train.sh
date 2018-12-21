folder=$1
size=$2
srun -p gpu -w falcon3 --gres gpu:tesla:1 -t 11520 --mem 8G -c 4 -u python -X faulthandler ../pix2pix-tensorflow/pix2pix.py --mode train --output_dir ../data/ati/$folder/model$size --max_epochs 1000 --input_dir ../data/ati/$folder/combined256/ --which_direction AtoB --no_flip --checkpoint ../data/ati/$folder/model$size
