# HW3
How to train:
python3 train.py 
--train_data_path: Path to the training data.
--valid_data_path: Path to the validation data.
--max_len: max length.
--lr: learn rate.
--batch_size: batch size.
--num_epoch: epoch.
--model_type: seq2seq model type.
--model_dir: where to put checkpoints.
--log_dir: where to put logs.

How to test:
python3 test.py
--test_data_path: Path to the testing data.
--max_len: max length.
--num_beams: beam size for beam search.
--output_dir: directory to put outputs.
--model_type: seq2seq model type.
--model_path: checkpoint path.
--pred_file: File path to the prediction file.