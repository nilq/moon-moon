export clone_list = (tensor_list, zero_too) ->
  out = {}

  for k, v in pairs tensor_list
    out[k] = v\clone!
    out[k]\zero! if zero_too
  out
