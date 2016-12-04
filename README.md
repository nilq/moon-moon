# moon-moon
Character level model generation and sampling with LSTM, RNN, GRU etc. neural networks using Torch.

---

```
$ make train
```

```
$ moonc .
$ th train.lua -h
Built ./train.moon
Built ./model/lstm.moon
Built ./model/rnn.moon
Built ./model/init.moon
Built ./model/gru.moon
Built ./utils/model_utils.moon
Built ./utils/util.moon
Built ./utils/charsplit_lm_minibatch_loader.moon
Built ./utils/init.moon
Built ./utils/onehot.moon
Usage: /home/nilq/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th [options]

'Moon Moon' by Nilq

Options
  -data_dir                  data directory should contain the file 'input.txt' with input data [data/rrmartin]
  -rnn_size                  size of LSTM internal states [128]
  -num_layers                number of layers in LSTM [2]
  -model                     lstm, gru or rnn [lstm]
  -learning_rate             learning rate of the model [0.002]
  -learning_rate_decay       learning rate decay of model [0.97]
  -learning_rate_decay_after number of epochs before learning rate [10]
  -decay_rate                decay rate for rmsprop [0.95]
  -dropout                   dropout for regularization, used after each RNN hidden layer [0]
  -seq_length                number of timesteps to unroll network for [50]
  -batch_size                number of sequences to train on in parallel [50]
  -max_epochs                number of full passes through the training data [50]
  -grad_clip                 clip gradients at this value [5]
  -train_frac                fraction of data going into training set [0.95]
  -val_frac                  fraction of data going into validation set [0.05]
  -init_from                 initialize network parameters from checkpoint at this path []
  -seed                      torch manual random number generation seed [1337]
  -print_every               amount of minibatches between status reports of loss [1]
  -eval_val_every            amount of iterations between evaluation of validation data [1000]
  -checkpoint_dir            output directory where checkpoints get written [cv]
  -savefile                  path to autosave checkpoint to; will be inside 'checkpoint_dir/' [lstm]
  -accurate_gpu_timing       0: 'hell nah', 1: 'yes, please do waste a lot of power getting precise GPU timing <3' [0]
  -gpuid                     which GPU to use; -1 being 'please use CPU' [0]
  -opencl                    use OpenCL rather than CUDA [0]

```

---

"Skud ud til Karpathy"
