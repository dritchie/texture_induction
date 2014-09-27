local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")


local DecolorizeNode = node.makeNodeFromFunc("DecolorizeNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	return terra(input: RGBAColor)
		-- Convert color to luma
		return 0.3*input(0) + 0.59*input(1) + 0.11*input(2)
	end, {4}
end)


local Decolorize = S.memoize(function(real, GPU)

	local BaseFunction = Function(real, 4, GPU)
	local Decolorize = BaseFunction.makeDefaultSubtype(
	"Decolorize",
	{
		{input = 4}
	},
	{})

	terra Decolorize:expand(coordNode: &Decolorize.CoordNode) : &Decolorize.OutputNode
		return [DecolorizeNode(real, GPU)].alloc():init(self.registers, self.input:expand(coordNode))
	end
	inherit.virtual(Decolorize, "expand")

	return Decolorize

end)


return
{
	Decolorize = Decolorize
}