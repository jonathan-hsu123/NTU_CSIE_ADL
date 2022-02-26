# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```

maybe good:
Max_len 128, hid 512, num_layer 2, drop_out 0.2, !bi, lr 1e-3, batch size 32
test1:
Max_len 128, hid 512, num_layer 2, drop_out 0.3, !bi, lr 1e-3, batch size 32 layer_norm
strong:
Max_len 128, hid 512, num_layer 2, drop_out 0.3, bi, lr 1e-3, batch size 128 layer_norm 
lr_5e-4:
Max_len 128, hid 512, num_layer 2, drop_out 0.3, bi, lr 5e-4, batch size 128 layer_norm 
# NTU_CSIE_ADL
Application of deep learning 
