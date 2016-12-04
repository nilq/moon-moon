OneHot, parent = torch.class "OneHot", "nn.Module"

OneHot.__init = (@output_size) =>
  parent.__init @

  @_eye = torch.eye output_size

OneHot.update_output = (input) =>
  (@output\resize (input\size 1), @output_size)\zero!

  if @_eye == nil
    @_eye = torch.eye @output_size

  @_eye = @_eye\float!

  long_inut = input\long!

  @output\copy @_eye\index 1, long_inut

  @output

OneHot
