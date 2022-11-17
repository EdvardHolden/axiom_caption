ID_FILE="data/deepmath/val.txt"
python3 -u train.py --model_dir experiments/sandbox/conjecture/ --problem_features data/raw/deepmath_conjectures.pkl --train_id_file $ID_FILE --val_id_file $ID_FILE
