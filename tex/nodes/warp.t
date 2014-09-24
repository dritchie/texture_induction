local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.nodes.node")
local Vec = terralib.require("utils.linalg.vec")
local finitediff = terralib.require("tex.nodes.finitediff")


-- Transforms the input coordinate by the discrete derivative of its second
--    grayscale input
-- Implementation constructs a hidden sub-graph to compute the derivatives.


local DELTA = 1.0/1024


local WarpNode = node.makeNodeFromFunc(function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local Vec1 = Vec(real, 1, GPU)
	return terra(coord: Coord, input: Vec1, inputdx: Vec1, inputdy: Vec1, strength: real) : Coord
		var dx = ((inputdx - input) / DELTA) * strength
		var dy = ((inputdy - input) / DELTA) * strength
		coord(0) = coord(0) + dx
		coord(1) = coord(1) + dy
		return coord
	end
end, {1, 1, 1})


local WarpNodeWrapper = S.memoize(function(real, GPU)
	local WarpNodeT = WarpNode(real, GPU)
	local rawCreate = WarpNodeT.methods.create
	local succ, T = rawCreate:peektype()
	local Registers = T.parameters[1]
	local CoordNode = T.parameters[2]
	local GrayscaleNode = T.parameters[3]
	WarpNodeT.methods.create = terra(registers: Registers, coordNode: CoordNode, inputNode: GrayscaleNode, strength: real)
		var dxNode = [finitediff.XShiftNode(real, GPU)].create(registers, coordNode, DELTA)
		var dyNode = [finitediff.YShiftNode(real, GPU)].create(registers, coordNode, DELTA)
		return rawCreate(registers, coordNode, inputNode, dxNode, dyNode, strength)
	end
	return WarpNodeT
end)


return
{
	WarpNode = WarpNodeWrapper
}
