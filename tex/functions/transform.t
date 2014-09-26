local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Mat = terralib.require("utils.linalg.mat")


local TransformNode = node.makeNodeFromFunc("TransformNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local Mat3 = Mat(real, 3, 3, GPU)
	return terra(coord: Coord, xform: Mat3)
		return xform:transformPoint(coord)
	end, {2}
end)


return
{
	TransformNode = TransformNode
}
