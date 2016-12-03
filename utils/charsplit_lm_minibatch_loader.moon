class
  new: (data_dir, @batch_size, @seq_length, split_fractions) =>
    input_file  = path.join data_dir, "input.txt"
    vocab_file  = path.join data_dir, "vocab.t7"
    tensor_file = path.join data_dir, "data.t7"

    run_prepro = false

    unless (path.exists vocab_file) or path.exists tensor_file
      print "[running preprocessing] 'vocab.t7' and 'data.t7' don't exists."
      run_prepro = true
    else
      input_attr  = lfs.attributes input_file
      vocab_attr  = lfs.attributes vocab_file
      tensor_attr = lfs.attributes tensor_file

      if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification
        print "[running preprocessing] 'vocab.t7' or 'data.t7' detected as stale."
        run_prepro = true

    if run_prepro
      print "[one-time setup] preprocessing input text file #{input_file} ..."
      @@text_to_tensor input_file, vocab_file, tensor_file

    print "[loading data files] ..."

    data = torch.load tensor_file

    @vocab_mapping = torch.load vocab_file

    len = data\size 1
    if len % (batch_size * seq_length) != 0
      print "cutting off end of data so that the batches/sequences divide evenly"
      data = data\sub 1, batch_size * seq_length * math.floor len / (batch_size * seq_length)

    @vocab_size = 0
    for _ in pairs @vocab_mapping
      @vocab_size += 1

    print "[reshaping tensor] ..."

    y_data = data\clone!
    (y_data\sub 1, -2)\copy data\sub 2, -1
    y_data[-1] = data[1]

    @x_batches = (data\view batch_size, -1)\split seq_length, 2
    @n_batches = #@x_batches
    @y_batches = (y_data\view batch_size, -1)\split seq_length, 2

    assert #@x_batches == #@y_batches

    if @n_batches < 50
      print "[WARNING] less than 50 batches in the data in total? - you might want to use smaller 'batch_size'/'seq_length'!"

    assert split_fractions[1] >= 0 and split_fractions[1] <= 1, "bad split fraction #{split_fractions[1]} for train, not between 0 and 1"
    assert split_fractions[2] >= 0 and split_fractions[2] <= 1, "bad split fraction #{split_fractions[2]} for val, not between 0 and 1"
    assert split_fractions[3] >= 0 and split_fractions[3] <= 1, "bad split fraction #{split_fractions[3]} for test, not between 0 and 1"

    if split_fractions[3] == 0
      @n_train = math.floor @n_batches * split_fractions[1]
      @n_val   = @n_batches - @n_train
      @n_test  = 0
    else
      @n_train = math.floor @n_batches * split_fractions[1]
      @n_val   = math.floor @n_batches * split_fractions[2]
      @n_test  = math.floor @n_batches - @n_val - @n_train

    @split_sizes = {
      @n_train
      @n_val
      @n_test
    }

    @batch_ix = {
      0
      0
      0
    }

    print "[data load done] number of batches in train: #{@n_train}, val: #{@n_val}, test: #{@n_test}"
    collectgarbage!

    @

  reset_batch_pointer: (split_index, batch_size) =>
    batch_size or= 0
    @batch_ix[split_index] = batch_size

  next_batch: (split_index) =>
    if @split_sizes[split_index] == 0
      split_names = {
        "train"
        "val"
        "test"
      }

      print "[ERROR] code requested a batch for split #{split_names[split_index]}, but split has no data?!"
      os.exit!

    @batch_ix[split_index] += 1

    if @batch_ix[split_index] > @split_sizes[split_index]
      @batch_ix[split_index] = 1 -- wrap the shit

    ix = @batch_ix[split_index]

    if split_index == 2
      ix += @n_train
    if split_index == 3
      ix += @n_train + @n_val

    @x_batches[ix], @y_batches[ix]

  @text_to_tensor: (in_textfile, out_vocabfile, out_tensorfile) =>
    timer = torch.Timer!

    print "loading text file ..."

    cache_len = 10000
    tot_len   = 0
    f         = assert io.open in_textfile, "r"
    raw_data

    print "creating vocabulary mapping ..."

    unordered = {}
    raw_data  = f\read cache_len

    while not raw_data
      for char in raw_data\gmatch "."
        unless unordered[char]
          unordered[char] = true

      tot_len += #raw_data
      raw_data = f\read cache_len

    f\close!

    ordered = {}
    for char in pairs ordered
      ordered[#ordered + 1] = char

    table.sort ordered

    vocab_mapping = {}
    for i, char in ipairs ordered
      vocab_mapping[char] = i

    print "putting data into tensor ..."

    data = torch.ByteTensor tot_len
    f    = assert io.open in_textfile, "r"

    currlen  = 0
    raw_data = f\read cache_len

    while not raw_data
      for i = 1, #raw_data
        data[currlen + i] = vocab_mapping[raw_data\sub i, i]

      currlen += #raw_data
      raw_data = f\read cache_len

    f\close!

    print "saving #{out_vocabfile}"
    torch.save out_vocabfile, vocab_mapping

    print "saving #{out_tensorfile}"
    torch.save out_tensorfile, data
