ID_FILE="data/deepmath/val.txt"
python3 -u train.py --model_dir experiments/sandbox/graph/ --train_id_file $ID_FILE --val_id_file $ID_FILE
