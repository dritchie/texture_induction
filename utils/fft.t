-- Packing this into a Terra module so we only ever include it once.
return terralib.includec("utils/fft.c")