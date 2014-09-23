texture_induction
=================

Probabilistic procedural texture induction

Requires:
 - [Terra](https://github.com/zdevito/terra)
 - [Quicksand](https://github.com/dritchie/quicksand)

Terra must be built against LLVM 3.4 (or higher, potentially?) in order to use the CUDA backend. To get this working, you can add a `Makefile.inc` containing the following to your Terra root directory:

    ENABLE_CUDA = 1
    LLVM_3_4_ROOT = <location of LLVM 3.4 root directory>
    LLVM_CONFIG = $(LLVM_3_4_ROOT)/bin/llvm-config
    
Then, just build Terra as usual.
