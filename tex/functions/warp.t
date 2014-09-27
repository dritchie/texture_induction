local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")
local Derivative = terralib.require("tex.functions.derivative").Derivative


local WarpNode = node.makeNodeFromFunc("WarpNode", function(real, GPU)
	local Vec2 = Vec(real, 2, GPU)
	return terra(coord: Vec2, inputDeriv: Vec2, strength: real)
		return coord + strength*inputDeriv
	end, {2, 2}
end)


local Warp = S.memoize(function(real, nchannels, GPU)

	local BaseFunction = Function(real, nchannels, GPU)
	local Warp = BaseFunction.makeDefaultSubtype(
	"Warp",
	{
		{input = 1},
		{warpField = 1}
	},
	{
		{strength = real}
	})

	terra Warp:expand(coordNode: &Warp.CoordNode) : &Warp.OutputNode
		var derivFn = [Derivative(real, GPU)].alloc():init(self.registers, self.warpField)
		var derivNode = derivFn:expand(coordNode)
		-- Delete temporary derivFn *without* deleting the functions beneath it
		-- (Kind of hackish; would be more proper to persist derivFn as a member)
		derivFn.input = nil
		derivFn:delete()
		var warpNode = [WarpNode(real, GPU)].alloc():init(self.registers, coordNode, derivNode, self.strength)
		return self.input:expand(warpNode)
	end
	inherit.virtual(Warp, "expand")

	return Warp

end)


return
{
	Warp = Warp
}
