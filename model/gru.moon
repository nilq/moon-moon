gru = (input_size, rnn_size, n, dropout=0) ->
  inputs = {}

  table.insert inputs, nn.Identity!!
  for L = 1, n
    table.insert inputs, nn.Identity!!

  new_input_sum = (in_size, xv, hv) ->
    i2h = (nn.Linear in_size, rnn_size)  xv
    h2h = (nn.Linear rnn_size, rnn_size) hv

    nn.CAddTable! {i2h, h2h}

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

    ----------------------------------
    -- forward update and reset gates
    ----------------------------------
    update_gate = nn.Sigmoid! new_input_sum input_size_L, x, prev_h
    reset_gate  = nn.Sigmoid! new_input_sum input_size_L, x, prev_h

    ----------------------------------
    -- compute candidate hidden state
    ----------------------------------
    gated_hidden = nn.CMulTable! {reset_gate, prev_h}

    p2 = (nn.Linear rnn_size, rnn_size) gated_hidden
    p1 = (nn.Linear input_size_L, rnn_size) x

    hidden_candidate = nn.Tanh! nn.CAddTable! {p1, p2}

    ----------------------------------
    -- compute interpolated hidden state, based on 'update_gate'
    ----------------------------------
    zh     = nn.CMulTable! {update_gate, hidden_candidate}
    zhm1   = nn.CMulTable! {
      ((nn.AddConstant 1, false) nn.MulConstant -1, false) update_gate
      prev_h
    }
    next_h = nn.CAddTable {zh, zhm1}

    table.insert outputs, next_h

  top_h = outputs[#outputs]

  if dropout > 0
    top_h = (nn.Dropout dropout) top_h

    proj     = (nn.Linear rnn_size, input_size) top_h
    log_soft = nn.LogSoftMax! proj

    table.insert outputs, log_soft

    nn.gModule inputs, outputs

{
  :gru
}
