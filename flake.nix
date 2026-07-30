{
  description = "my_softmax_function — fused softmax-attention kernel";

  inputs = {
    # New merged repo (kernel-builder folded into huggingface/kernels).
    # Its metadata schema matches the current `kernels` loader library.
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    # genFlakeOutputs was renamed to genKernelFlakeOutputs in the new repo.
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
