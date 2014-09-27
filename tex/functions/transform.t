local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Mat = terralib.require("utils.linalg.mat")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")


local TransformNode = node.makeNodeFromFunc("TransformNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local Mat3 = Mat(real, 3, 3, GPU)
	return terra(coord: Coord, xform: Mat3)
		return xform:transformPoint(coord)
	end, {2}
end)


local Transform = S.memoize(function(real, nchannels, GPU)

	local BaseFunction = Function(real, nchannels, GPU)
	local Mat3 = Mat(real, 3, 3, GPU)
	local Transform = BaseFunction.makeDefaultSubtype(
	"Transform",
	{
		{input = nchannels}
	},
	{
		{xform = Mat3}
	})

	terra Transform:expand(coordNode: &Transform.CoordNode) : &Transform.OutputNode
		var tnode = [TransformNode(real, GPU)].alloc():init(self.registers, coordNode, self.xform)
		return self.input:expand(tnode)
	end
	inherit.virtual(Transform, "expand")

	return Transform

end)


return
{
	Transform = Transform
}
