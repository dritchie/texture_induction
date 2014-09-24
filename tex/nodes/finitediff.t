local node = terralib.require("tex.nodes.node")
local Vec = terralib.require("utils.linalg.vec")


-- There are a few operations that require discrete derivatives. We implement this by
--    having those operations explicitly declare those derivatives (i.e. the endpoints
--    of forward finite differences) as inputs. To compute these inputs, we perturb the
--    input coordinates along x and y according to a finite difference rule. The two
--    nodes defined in this file just perform that coordinate perturbation. 


local XShiftNode = node.makeNodeFromFunc(function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	return terra(coord: Coord, shiftamt: real)
		coord(0) = coord(0) + shiftamt
		return coord	
	end
end, {})


local YShiftNode = node.makeNodeFromFunc(function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	return terra(coord: Coord, shiftamt: real)
		coord(1) = coord(1) + shiftamt
		return coord	
	end
end, {})



return
{
	XShiftNode = XShiftNode,
	YShiftNode = YShiftNode
}

