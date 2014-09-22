local S = terralib.require("qs.lib.std")
local ImagePool = terralib.require("tex.imagePool")


-- Image 'registers' for storing vector interpreter intermediates
local Registers = S.memoize(function(real, GPU)

	local struct Registers(S.Object)
	{
		grayscaleRegisters: ImagePool(real, 1, GPU),
		colorRegisters: ImagePool(real, 4, GPU),
		coordinateRegisters: ImagePool(real, 2, GPU)
	}

	return Registers

end)


return Registers