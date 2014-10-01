local S = terralib.require("qs.lib.std")

local U = {}


U.printTabs = terra(n: uint)
	for i=0,n do S.printf("    ") end
end


return U