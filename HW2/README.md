The following steps is used to train context selection:
python3 train_select_context.py
These are the arguments for the program:
--context_path: Path to the context file.
--train_data_path: Path to the training data(input question).
--valid_data_path: Path to the validation data(input question).
--max_len: max length.
--lr: learn rate.
--batch_size: batch size.
--num_epoch: epoch.
--model_type: bert-based model type.
--model_dir: where to put checkpoints.
--log_dir: where to put logs.

The following steps is used to train QA:
python3 train_QA.py
These are the arguments for the program:
--context_path: Path to the context file.
--train_data_path: Path to the training data(input question).
--valid_data_path: Path to the validation data(input question).
--max_len: max length.
--lr: learn rate.
--batch_size: batch size.
--num_epoch: epoch.
--model_type: bert-based model type.
--model_dir: where to put checkpoints.
--log_dir: where to put logs.

The following steps is used to test QA result:
python3 test.py
These are the arguments for the program:
--context_path: Path to the context file.
--test_data_path: Path to the testing data(input question).
--max_len: max length.
--output_dir: directory to put outputs.
--model_type: bert-based model type.
--pred_file: File path to the prediction file.


