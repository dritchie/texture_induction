local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")


local DELTA = 1.0/1024


local XShiftNode = node.makeNodeFromFunc("XShiftNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	return terra(coord: Coord)
		coord(0) = coord(0) + DELTA
		return coord
	end, {2}
end)


local YShiftNode = node.makeNodeFromFunc("YShiftNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	return terra(coord: Coord)
		coord(1) = coord(1) + DELTA
		return coord	
	end, {2}
end)


local DerivativeNode = node.makeNodeFromFunc("DerivativeNode", function(real, GPU)
	local Vec1 = Vec(real, 1, GPU)
	local Vec2 = Vec(real, 2, GPU)
	return terra(input: Vec1, inputdx: Vec1, inputdy: Vec1)
		var dx = ((inputdx(0) - input(0)) / DELTA)
		var dy = ((inputdy(0) - input(0)) / DELTA)
		return Vec2.create(dx, dy)
	end, {1, 1, 1}
end)


local Derivative = S.memoize(function(real, GPU)

	local BaseFunction = Function(real, 2, GPU)
	local Derivative = BaseFunction.makeDefaultSubtype(
	"Derivative",
	{
		{input = 1}
	},
	{
	})

	terra Derivative:expand(coordNode: &Derivative.CoordNode) : &Derivative.OutputNode
		var xshift = [XShiftNode(real, GPU)].alloc():init(self.registers, coordNode)
		var yshift = [YShiftNode(real, GPU)].alloc():init(self.registers, coordNode)
		var input = self.input:expand(coordNode)
		var inputdx = self.input:expand(xshift)
		var inputdy = self.input:expand(yshift)
		return [DerivativeNode(real, GPU)].alloc():init(self.registers, input, inputdx, inputdy)
	end
	inherit.virtual(Derivative, "expand")

	return Derivative

end)


return
{
	Derivative = Derivative
}


