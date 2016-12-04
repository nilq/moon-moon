require "torch"
require "nn"
require "nngraph"
require "optim"
require "lfs"

require "util/onehot"
require "util/util"

cmd = torch.CmdLine!

cmd\text!
cmd\text!
cmd\text "'Moon Moon' by Nilq"
cmd\text!
cmd\text "Use checkpoint progress to write things!"
cmd\text!
cmd\text "Options"

cmd\argument "-model", "model checkpoint to use"

cmd\option "-seed", 1337, "torch manual random number generation seed"
cmd\option "-sample", 1, "0: 'optimally sample each iteration', 1: 'every single iteration'"
cmd\option "-prime_text", "", "base sample on this 'seed' for the LSTM"
cmd\option "-length", 2000, "length of sample in characters"
cmd\option "-temperature", 1, "temperature of sampler"
cmd\option "-gpuid", 0, "0<: 'which GPU to use', -1: 'use CPU'"
cmd\option "-opencl", 0, "use OpenCL rather than CUDA"
cmd\option "-verbose", 1, "0: 'very shh. much quiet. wow.', 1: 'print diagnostics'"

cmd\text!

opt = cmd\parse arg

gprint = (str) ->
  print str if opt.verbose == 1

if opt.gpuid >= 0
  if opt.opencl == 0
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
  else if opt.opencl == 1
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

torch.manualSeed opt.seed

unless lfs.attributes opt.model, "mode"
  gprint "[ERROR] File #{opt.model} is nowhere to be found!"

checkpoint = torch.load opt.model
protos     = checkpoint.protos

protos.rnn\evaluate!

----------------------------------
-- initialize vocabulary
----------------------------------
vocab  = checkpoint.vocab
ivocab = {}

for c, i in pairs vocab
  ivocab[i] = c
----------------------------------
-- initialize the RNN state
----------------------------------
gprint "creating a #{checkpoint.opt.model} ..."

current_state = {}
for L = 1, checkpoint.opt.num_layers
  h_init = (torch.zeros 1, checkpoint.opt.rnn_size)\double!

  if opt.gpuid >= 0
    h_init = h_init\cuda! if opt.opencl == 0
    h_init = h_init\cl!   if opt.opencl == 1

  table.insert current_state, h_init\clone!

  if checkpoint.opt.model == "lstm"
    table.insert current_state, h_init\clone!

state_size = #current_state

unless opt.prime_text == ""
  gprint "seeding LSTM memory with\n#{prime_text}"
  gprint "----------------------------------"

  for c in opt.prime_text\gmatch "."
    prev_char = torch.Tensor {vocab[c]}

    io.write ivocab[prev_char[1]]

    if opt.gpuid >= 0
      prev_char = prev_char\cuda! if opt.opencl == 0
      prev_char = prev_char\cl!   if opt.opencl == 1

    lst = protos.rnn\forward {
      prev_char
      unpack current_state
    }

    current_state = {}

    for i = 1, state_size
      table.insert current_state, lst[i]

    export prediction = lst[#lst]
else
  gprint "no seeding text, use uniform probability over first character"
  gprint "----------------------------------"

  export prediction = ((torch.Tensor 1, #ivocab)\fill 1) / #ivocab

  if opt.gpuid >= 0
    prediction = prediction\cuda! if opt.opencl == 0
    prediction = prediction\cl!   if opt.opencl == 1

for i = 1, opt.length
  if opt.sample == 0
    _, _prev_char = prediction\max 2
    export prev_char           = _prev_char\resize 1
  else
    prediction\div opt.temperature -- scale by temperature

    probs = (torch.exp prediction)\squeeze!
    probs\div torch.sum probs

    export prev_char = ((torch.multinomial probs\float!, 1)\resize 1)\float!

  lst = protos.rnn\forward {
    prev_char
    unpack current_state
  }

  current_state = {}

  for i = 1, state_size
    table.insert current_state, lst[i]

  prediction = lst[#lst]

  io.write ivocab[prev_char[1]]

io.write "\n"
io.flush!
