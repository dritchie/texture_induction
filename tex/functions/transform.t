local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Mat = terralib.require("utils.linalg.mat")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")
local util = terralib.require("utils.util")


local TransformNode = node.makeNodeFromFunc("TransformNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local Mat3 = Mat(real, 3, 3, GPU)
	return terra(coord: Coord, xform: Mat3)
		return xform:transformPoint(coord)
	end, {2}
end)


local Transform = S.memoize(function(real, nchannels, GPU)

	local BaseFunction = Function(real, nchannels, GPU)
	local Mat3_cpu = Mat(real, 3, 3, false)
	local Mat3 = Mat(real, 3, 3, GPU)
	local Transform = BaseFunction.makeDefaultSubtype(
	"Transform",
	{
		{input = nchannels}
	},
	{
		{xform = Mat3_cpu}
	})

	terra Transform:expand(coordNode: &Transform.CoordNode) : &Transform.OutputNode
		-- coerce CPU matrix into platform-specific matrix
		var tnode = [TransformNode(real, GPU)].alloc():init(self.registers, coordNode, @([&Mat3](&self.xform)))
		return self.input:expand(tnode)
	end
	inherit.virtual(Transform, "expand")

	terra Transform:printAggParams(tablevel: uint) : {}
		util.printTabs(tablevel)
		S.printf("xform:\n")

		util.printTabs(tablevel+1)
		S.printf("%.2g %.2g %.2g\n", self.xform(0,0), self.xform(0,1), self.xform(0,2))
		util.printTabs(tablevel+1)
		S.printf("%.2g %.2g %.2g\n", self.xform(1,0), self.xform(1,1), self.xform(1,2))
		util.printTabs(tablevel+1)
		S.printf("%.2g %.2g %.2g\n", self.xform(2,0), self.xform(2,1), self.xform(2,2))
	end
	inherit.virtual(Transform, "printAggParams")

	return Transform

end)


return
{
	Transform = Transform
}
