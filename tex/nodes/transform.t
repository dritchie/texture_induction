local node = terralib.require("tex.nodes.node")
local Vec = terralib.require("utils.linalg.vec")
local Mat = terralib.require("utils.linalg.mat")


local TransformNode = node.makeNodeFromFunc(function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local Mat3 = Mat(real, 3, 3, GPU)
	return terra(coord: Coord, xform: Mat3)
		return xform:transformPoint(coord)
	end
end, {})


return
{
	TransformNode = TransformNode
}
