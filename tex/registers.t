local S = require("qs.lib.std")
local ImagePool = require("tex.imagePool")


-- Image 'registers' for storing vector interpreter intermediates
local Registers = S.memoize(function(real, GPU)

	local struct Registers(S.Object)
	{
		vec1Registers: ImagePool(real, 1, GPU),
		vec2Registers: ImagePool(real, 2, GPU),
		vec3Registers: ImagePool(real, 3, GPU),
		vec4Registers: ImagePool(real, 4, GPU)
	}

	return Registers

end)


return Registers
