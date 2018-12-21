folder=$1
size=$2
scp "rocket.hpc.ut.ee:christmas2018/data/ati/$folder/model${size}_reduced/frozen_model.pb" $folder.pb
