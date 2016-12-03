rnn = (input_size, rnn_size, n, dropout) ->
  inputs = {}

  table.insert inputs, nn.Identity!!
  for L = 1, n
    table.insert inputs, nn.Identity!!

  x, input_size_L
  outputs = {}

  for L = 1, n
    prev_h = inputs[L + 1]

    if L == 1
      x = (OneHot input_size) inputs[1]
      input_size_L = input_size
    else
      x = outputs[L - 1]

      if dropout > 0
        x = (nn.Dropout dropout) x

      input_size_L = rnn_size

    i2h    = (nn.Linear input_size_L, rnn_size) x
    h2h    = (nn.Linear rnn_size, rnn_size) prev_h
    next_h = nn.Tanh! nn.CAddTable! {i2h, h2h}

    table.insert outputs, next_h

  top_h = outputs[#outputs]

  top_h    = (nn.Dropout dropout) top_h if dropout > 0
  proj     = (nn.Linear rnn_size, input_size) top_h
  log_soft = nn.LogSoftMax! proj

  table.insert outputs, log_soft

  nn.gModule inputs, outputs

{
  :rnn
}
