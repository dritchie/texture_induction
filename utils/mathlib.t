local tmath = require("qs.lib.tmath")
local cumath = require("utils.cuda.cumath")

return function(GPU)
	assert(GPU ~= nil, "mathlib: GPU flag must be explicitly provided")
	if GPU then
		return cumath
	else
		return tmath
	end
end
