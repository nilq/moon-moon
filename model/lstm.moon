lstm = (input_size, rnn_size, n, dropout=0) ->
  inputs = {}

  table.insert inputs, nn.Identity!!
  for L = 1, n
    table.insert inputs, nn.Identity!!
    table.insert inputs, nn.Identity!!

  local x, input_size_L
  outputs = {}

  for L = 1, n
    prev_h = inputs[L * 2 + 1]
    prev_c = inputs[L * 2]

    if L == 1
      x = (OneHot input_size) inputs[1]
      input_size_L = input_size
    else
      x = outputs[(L - 1) * 2]
      x = (nn.Dropout dropout) x if dropout > 0

      input_size_L = rnn_size

    i2h = ((nn.Linear input_size_L, 4 * rnn_size) x)\annotate {name: "i2h_#{L}"}
    h2h = ((nn.Linear rnn_size, 4 * rnn_size) prev_h)\annotate {name: "h2h_#{L}"}

    all_input_sums = nn.CAddTable! {i2h, h2h}

    reshaped = (nn.Reshape 4, rnn_size) all_input_sums
    n1, n2, n3, n4 = ((nn.SplitTable 2) reshaped)\split 4

    ----------------------------------
    -- decode all the gates!
    ----------------------------------
    in_gate     = nn.Sigmoid! n1
    forget_gate = nn.Sigmoid! n2
    out_gate    = nn.Sigmoid! n3
    ----------------------------------
    -- decode the write inputs
    ----------------------------------
    in_transform = nn.Tanh! n4
    ----------------------------------
    -- perfon long-short-term-memory update
    ----------------------------------
    next_c = nn.CAddTable! {
      nn.CMulTable! {forget_gate, prev_c}
      nn.CMulTable! {in_gate,     in_transform}
    }

    next_h = nn.CMulTable! {
      out_gate
      nn.Tanh! next_c
    }

    table.insert outputs, next_c
    table.insert outputs, next_h

  top_h = outputs[#outputs]
  top_h = (nn.Dropout dropout) top_h if dropout > 0

  proj  = ((nn.Linear rnn_size, input_size) top_h)\annotate {name: "decoder"}

  log_soft = nn.LogSoftMax! proj

  table.insert outputs, log_soft

  nn.gModule inputs, outputs

{
  :lstm
}
