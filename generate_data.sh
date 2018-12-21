mkdir -p $1
cd $1
python generate_train_data.py --file ../$1.mp4
cd ..
