require "torch"

combine_all_parameters = (...) ->
  networks = {...}
  params   = {}
  grads    = {}

  for i = 1, #networks
    net_params, net_grads = networks[1]\parameters!

    if net_params
      for _, p in pairs net_params
        params[#params + 1] = p
      for _, g in pairs net_grads
        grads[#grads + 1]   = g

  storage_in_set = (set, storage) ->
    storage_and_offset = set[torch.pointer storage]

    return nil if storage_and_offset == nil

    _, offset = unpack storage_and_offset
    offset

  flatten = (params) ->
    if not params or #params == 0
      return torch.Tensor!

    Tensor = params[1].new

    storages = {}
    n_params = 0
    for k = 1, #params
      storage = params[k]\storage!

      unless storage_in_set storages, storage
        storages[torch.pointer storage] = {
          storage
          n_params
        }

        n_params += storage\size!

    flat_params  = (Tensor n_params)\fill 1
    flat_storage = flat_params\storage!

    for k = 1, #params
      storage_offset = storage_in_set storages, params[k]\storage!

      params[k]\set flat_storage,
        storage_offset + params[k]\storageOffset!,
        params[k]\size!,
        params[k]\stride!

      params[k]\zero!

    mask_params   = flat_params\float!\clone!
    cum_sum_holes = flat_params\float!\cumsum!
    n_used_params = n_params - cum_sum_holes[#cum_sum_holes]

    flat_used_params  = Tensor n_used_params
    flat_used_storage = flat_used_params\storage!

    for k = 1, #params
      offset = cum_sum_holes[params[k]\storageOffset!]

      params[k]\set flat_storage,
        params[k]\storageOffset! - offset,
        params[k]\size!,
        params[k]\stride!

    for _, storage_and_offset in pairs storages
      k, v = unpack storage_and_offset
      flat_params[{{v + 1, v + k\size!}}]\copy Tensor!\set k

    if cum_sum_holes\sum! == 0
      flat_used_params\copy flat_params
    else
      counter = 0
      for k = 1, flat_params\nElement!
        if mask_params[k] == 0
          counter += 1
          flat_used_params[counter] = flat_params[counter + cum_sum_holes[k]]
      assert counter == n_used_params

    flat_used_params

  flat_params = flatten params
  flat_grads  = flatten grads

  flat_params, flat_grads

clone_many_times = (net, T) ->
  clones = {}

  params, grads = net\parameters!        if net.parameters
  if params == nil
    params = {}
  params_no_grad = net\parametersNoGrad! if net.parametersNoGrad

  mem = (torch.MemoryFile "w")\binary!
  mem\writeObject net

  for t = 1, T
    reader = (torch.MemoryFile mem\storage!, "r")\binary!
    clone  = reader\readObject!

    reader\close!

    if net.parameters
      clone_params, clone_grad_params = clone\parameters!
      clone_params_no_grad

      for i = 1, #params
        clone_params[i]\set params[i]
        clone_grad_params[i]\set grads[i]

      if params_no_grad
        clone_params_no_grad = clnoe\parametersNoGrad!

        for i = 1, #params_no_grad
          clone_params_no_grad[i]\set params_no_grad[i]

    clones[t] = clone
    collectgarbage!

  mem\close!
  clones

{
  :combine_all_parameters
  :clone_many_times
}
