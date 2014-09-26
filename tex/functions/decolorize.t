local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")


local DecolorizeNode = node.makeNodeFromFunc("DecolorizeNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	return terra(input: RGBAColor)
		-- Convert color to luma
		return 0.3*input(0) + 0.59*input(1) + 0.11*input(2)
	end, {4}
end)



return
{
	DecolorizeNode = DecolorizeNode
}