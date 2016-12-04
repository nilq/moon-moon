require "torch"
require "nn"
require "nngraph"

require "util/onehot"
require "util/util"

cmd = torch.CmdLine!

cmd\text!
cmd\text "'Moon Moon' by Nilq"
cmd\text!
cmd\text "Load checkpoint and inspect validation losses"
cmd\text!
cmd\text "Options"
cmd\argument "-model", "model to load"
cmd\option "-gpuid", 0, "GPU to use"
cmd\option "-opencl", 0, "use OpenCL rather than CUDA"
cmd\text!

opt = cmd\parse arg

if opt.gpuid >= 0
  if opt.opencl == 0
    print "using CUDA on GPU #{opt.gpuid}"

    require "cutorch"
    require "cunn"

    cutorch\setDevice opt.gpuid + 1

  if opt.opencl == 1
    print "using OpenCL on GPU #{opt.gpuid}"

    require "cltorch"
    require "clnn"

    cltorch\setDevice opt.gpuid + 1

model = torch.load opt.model

print "options:\n#{model.opt}"
print "validation losses:\n#{model.val_losses}"
