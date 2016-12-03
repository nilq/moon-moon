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

test_frac   = math.max 0, 1 - (opt.train_frac + opt.val_frac)
split_sizes = {
  opt.train_frac
  opt.val_frac
  opt.test_frac
}

if opt.gpuid >= 0 and opt.opencl == 0
  ok, cunn     = pcall require, "cunn"
  ok2, cutorch = pcall require, "cutorch"

  print "package CUNN not found!"    unless ok
  print "package CUTorch not found!" unless ok2

  if ok and ok2
    print "using CUDA on GPU #{opt.gpuid} ..."

    cutorch.setDevice  opt.gpuid + 1
    cutorch.manualSeed opt.seed
  else
    print "Fucked up some CUDA things ..."
    print "Falling back to CPU mode ..."
    opt.gpuid = -1

if opt.gpuid >= 0 and opt.opencl == 1
  ok, clnn     = pcall require, "clnn"
  ok2, cltorch = pcall require, "cltorch"

  print "package CLNN not found!"    unless ok
  print "package CLTorch not found!" unless ok2

  if ok and ok2
    print "using OpenCL on GPU #{opt.gpuid} ..."

    cltorch.setDevice opt.gpuid + 1
    torch.manualSeed  opt.seed
  else
    print "Fucked up some CL things ..."
    print "Falling back to CPU mode ..."
    opt.gpuid = -1

loader     = CharSplitLMMinibatchLoader opt.data_dir, opt.batch_size, opt.seq_length, split_sizes
vocab_size = loader.vocab_size
vocab      = loader.vocab_mapping

print "vocab size: #{vocab_size}"

unless path.exists opt.checkpoint_dir
  lfs.mkdir opt.checkpoint_dir

do_random_init = true
if 0 < string.len opt.init_from
  print "loading a model from checkpoint #{opt.init_from}"

  checkpoint    = torch.load opt.init_from
  export protos = checkpoint.protos

  vocab_compatible      = true
  checkpoint_vocab_size = 0

  for c, i in pairs checkpoint.vocab
    unless vocab[c] == i
      vocab_compatible = false

    checkpoint_vocab_size += 1

  unless checkpoint_vocab_size == vocab_size
    vocab_compatible = false
    print "checkpoint_vocab_size: #{checkpoint_vocab_size}"

  assert vocab_compatible, "[error] character vocabulary is all wrong!"

  print "overwriting parameters based on checkpoint ..."

  opt.rnn_size   = checkpoint.opt.rnn_size
  opt.num_layers = checkpoint.opt.num_layers
  opt.model      = checkpoint.opt.model

  do_random_init = false

else
  print "creating an #{opt.model} with #{opt.num_layers} layers ..."
  protos = {}

  switch opt.model
    when "lstm"
      protos.rnn = LSTM.lstm vocab_size, opt.rnn_size, opt.num_layers, opt.dropout
    when "gru"
      protos.rnn = GRU.gru   vocab_size, opt.rnn_size, opt.num_layers, opt.dropout
    when "rnn"
      protos.rnn = RNN.rnn   vocab_size, opt.rnn_size, opt.num_layers, opt.dropout

  protos.criterion = nn.ClassNLLCriterion!

init_state = {}
for L = 1, opt.num_layers
  h_init = torch.zeros opt.batch_size, opt.rnn_size

  if opt.gpuid >= 0
    h_init = h_init\cuda! if opt.opencl == 0
    h_init = h_init\cl!   if opt.opencl == 1

  table.insert init_state, h_init\clone!

  if opt.model == "lstm"
    table.insert init_state, h_init\clone!

if opt.gpuid >= 0
  if opt.opencl == 0
    for k, v in pairs protos
      v\cuda!
  elseif opt.opencl == 1
    for k, v in pairs protos
      v\cl!

params, grad_params = model_utils\combine_all_parameters protos.rnn
params\uniform -0.08, 0.08 if do_random_init

print "number of parameters in the model: #{params\nElement!}"

clones = {}
for name, proto in pairs protos
  print "cloning #{name}"
  clones[name] = model_utils.clone_many_times proto, opt.seq_length, not proto.parameters

export prepro = (x, y) ->
  x = (x\transpose 1, 2)\contiguous!
  y = (y\transpose 1, 2)\contiguous!

  if opt.gpuid >= 0
    if opt.opencl == 0
      x = x\float!\cuda!
      y = y\float!\cuda!
    elseif opt.opencl == 1
      x = x\cl!
      y = y\cl!

  x, y

export eval_split = (split_index, max_batches) ->
  print "evaluating loss over split index #{split_index}"

  n = loader.split_sizes[split_index]
  n = math.min max_batches, n if max_batches ~= nil

  loader\reset_batch_pointer split_index

  loss = 0
  rnn_state = {
    [0]: init_state
  }

  for i = 1, n
    x, y = loader\next_batch split_index
    x, y = prepro x, y

    for t = 1, opt.seq_length
      clones.rnn[t]\evaluate!

      lst = clones.rnn[t]\forward {
        x[t]
        unpack rnn_state[t - 1]
      }

      rnn_state[t] = {}

      for i = 1, #init_state
        table.insert rnn_state[t], lst[i]

      prediction = lst[#lst]
      loss      += clones.criterion[t]\forward prediction, y[t]

    loss /= opt.seq_length / n
    print "#{i} / #{n} ..."

    loss

init_state_global = clone_list init_state
export feval = (x) ->
  if x ~= params
    params\copy x

  grad_params\zero!

  ----------------------------------
  -- get minibatch
  ----------------------------------
  x, y = loader\next_batch 1
  x, y = prepro x, y
  ----------------------------------
  -- forward pass
  ----------------------------------
  loss       = 0
  predictions = {}
  rnn_state  = {
    [0]: init_state_global
  }

  for t = 1, opt.seq_length
    clones.rnn[t]\training!

    lst = clones.rnn[t]\forward {
      x[t]
      unpack rnn_state[t - 1]
    }

    rnn_state[t] = {}

    for i = 1, #init_state
      table.insert rnn_state[t], lst[i]

    predictions[t] = lst[#lst]

    loss += clones.criterion[t]\forward predictions[t], y[t]

  loss /= opt.seq_length
  ----------------------------------
  -- backward pass
  ----------------------------------
  drnn_state = {
    [opt.seq_length]: clone_list init_state, true
  }
  for t = opt.seq_length, 1, -1
    doutput_t = clones.criterion[t]\backward predictions[t], y[t]

    table.insert drnn_state[t], doutput_t

    dlst = clones.rnn[t]\backward {
      x[t]
      unpack rnn_state[t - 1]
      drnn_state[t]
    }

    drnn_state[t - 1] = {}

    for k, v in pairs dlst
      if k > 1
        drnn_state[t - 1][k - 1] = v

  ----------------------------------
  -- misc
  ----------------------------------
  init_state_global = rnn_state[#rnn_state]
  grad_params\clamp -opt.grad_clip, opt.grad_clip

  loss, grad_params

----------------------------------
-- optimization stuff
----------------------------------
train_losses = {}
val_losses   = {}

optim_state  = {
  learningRate: opt.learning_rate
  alpha: opt.decay_rate
}

iterations       = opt.max_epochs * loader.ntrain
iterations_epoch = loader.ntrain

loss0

for i = 1, iterations
  epoch = i / loader.ntrain

  timer   = torch.Timer!
  _, loss = optim.rmsprop feval, params, optim_state

  if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0
    cutorch.synchronize!

  train_losses = loss[1]

  if i % loader.ntrain == 0 and opt.learning_rate_decay < 1
    if epoch >= opt.learning_rate_decay_after
      optim_state.learningRate *= opt.learning_rate_decay

      print "decayed learning rate by coefficient #{opt.learning_rate_decay} to #{optim_state.learningRate}"

  if i % opt.eval_val_every == 0 or i == iterations
    val_loss   = eval_split 2
    val_losses = val_loss

    savefile = string.format "%s/lm_%s_epoch%.2f_%.4f.t7", opt.checkpoint_dir, opt.savefile, epoch, val_loss

    print "saving to checkpoint to #{savefile}"

    checkpoint = {
      :protos
      :opt
      :train_losses
      :val_loss
      :val_losses
      :i
      :epoch
      vocab: loader.vocab_mapping
    }

    torch.save savefile, checkpoint

  if i % opt.print_every == 0
    print string.format "%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params\norm! / params\norm!, time

  if i % 10 == 0
    collectgarbage!

  if loss[1] != loss[1]
    print "what the fuck? kill me please"

  if loss0 == nil
    loss0 = loss[1]

  if loss[1] > loss0 * 3
    print "loss is exploding, kill me please"
    break -- $ fucking halt
