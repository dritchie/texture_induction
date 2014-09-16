local tmath = terralib.require("qs.lib.tmath")
local cumath = terralib.require("utils.cumath")

return function(GPU)
	assert(GPU ~= nil, "mathlib: GPU flag must be explicitly provided")
	if GPU then
		return cumath
	else
		return tmath
	end
end