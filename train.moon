require "torch"
require "nn"
require "nngraph"
require "optim"
require "lfs"

import CharSplitLMMinibatchLoader, model_utils from require "utils"
import LSTM, GRU, RNN                          from require "model"

cmd = torch.CmdLine!
cmd\text!
cmd\text "'Moon Moon' by Nilq"
cmd\text!
cmd\text "Options"
----------------------------------
-- data stuff
----------------------------------
cmd\option "-data_dir", "data/rrmartin", "data directory should contain the file 'input.txt' with input data"
----------------------------------
-- model stuff
----------------------------------
cmd\option "-rnn_size", 128, "size of LSTM internal states"
cmd\option "-num_layers", 2, "number of layers in LSTM"
cmd\option "-model", "lstm", "lstm, gru or rnn"
----------------------------------
-- optimization stuff
----------------------------------
cmd\option "-learning_rate", 2e-3, "learning rate of the model"
cmd\option "-learning_rate_decay", 0.97, "learning rate decay of model"
cmd\option "-learning_rate_decay_after", 10, "number of epochs before learning rate"
cmd\option "-decay_rate", 0.95, "decay rate for rmsprop"
cmd\option "-dropout", 0, "dropout for regularization, used after each RNN hidden layer"
cmd\option "-seq_length", 50, "number of timesteps to unroll network for"
cmd\option "-batch_size", 50, "number of sequences to train on in parallel"
cmd\option "-max_epochs", 50, "number of full passes through the training data"
cmd\option "-grad_clip", 5, "clip gradients at this value"
cmd\option "-train_frac", 0.95, "fraction of data going into training set"
cmd\option "-val_frac", 0.05, "fraction of data going into validation set"

cmd\option "-init_from", "", "initialize network parameters from checkpoint at this path"
----------------------------------
-- book keeping stuff
----------------------------------
cmd\option "-seed", 1337, "torch manual random number generation seed"
cmd\option "-print_every", 1, "amount of minibatches between status reports of loss"
cmd\option "-eval_val_every", 1000, "amount of iterations between evaluation of validation data"
cmd\option "-checkpoint_dir", "cv", "output directory where checkpoints get written"
cmd\option "-savefile", "lstm", "path to autosave checkpoint to; will be inside 'checkpoint_dir/'"
cmd\option "-accurate_gpu_timing", 0, "0: 'hell nah', 1: 'yes, please do waste a lot of power getting precise GPU timing <3'"
----------------------------------
-- GPU/CPU
----------------------------------
cmd\option "-gpuid", 0, "which GPU to use; -1 being 'please use CPU'"
cmd\option "-opencl", 0, "use OpenCL rather than CUDA"
cmd\text!

export opt = cmd\parse arg

torch.manualSeed opt.seed
