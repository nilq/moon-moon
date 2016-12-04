require "torch"
require "nn"
require "nngraph"

require "util/onehot"
require "util/util"

cmd = torch.CmdLine!

cmd\text!
cmd\text "'Moon Moon' by Nilq"
cmd\text!
cmd\text "Load checkpoint and inspect validation losses!"
cmd\text!
cmd\text "Options"
cmd\argument "-model", "model to load"
cmd\option "-gpuid", 0, "GPU to use"
cmd\option "-opencl", 0, "use OpenCL rather than CUDA"
cmd\text!

opt = cmd\parse arg

if opt.gpuid >= 0
  if opt.opencl == 0
    ok, cunn     = pcall require, "cunn"
    ok2, cutorch = pcall require, "cutorch"

    print "package CUNN not found!"    unless ok
    print "package CUTorch not found!" unless ok2

    if ok and ok2
      print "using CUDA on GPU #{opt.gpuid} ..."

      cutorch.setDevice  opt.gpuid + 1
    else
      print "Fucked up some CUDA things ..."
  else if opt.opencl == 1
    ok, clnn     = pcall require, "clnn"
    ok2, cltorch = pcall require, "cltorch"

    print "package CLNN not found!"    unless ok
    print "package CLTorch not found!" unless ok2

    if ok and ok2
      print "using OpenCL on GPU #{opt.gpuid} ..."

      cltorch.setDevice opt.gpuid + 1
    else
      print "Fucked up some CL things ..."

model = torch.load opt.model

print "options:\n#{model.opt}"
print "validation losses:\n#{model.val_losses}"
