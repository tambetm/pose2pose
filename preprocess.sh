# Resize original images
folder=$1
size=$2
PIX2PIX_ROOT=../pix2pix-tensorflow

python $PIX2PIX_ROOT/tools/process.py \
  --input_dir $folder/original \
  --operation resize \
  --output_dir $folder/original_resized$size \
  --size $size

# Resize landmark images
python $PIX2PIX_ROOT/tools/process.py \
  --input_dir $folder/landmarks \
  --operation resize \
  --output_dir $folder/landmarks_resized$size \
  --size $size
  
# Combine both resized original and landmark images
python $PIX2PIX_ROOT/tools/process.py \
  --input_dir $folder/landmarks_resized$size \
  --b_dir $folder/original_resized$size \
  --operation combine \
  --output_dir $folder/combined$size
  
exit

# Split into train/val set
python $PIX2PIX_ROOT/tools/split.py \
  --dir $folder/combined$size

# Train the model on the data
python $PIX2PIX_ROOT/pix2pix.py \
  --mode train \
  --output_dir $folder/model \
  --max_epochs 10 \
  --input_dir $folder/combined/train \
  --which_direction AtoB
