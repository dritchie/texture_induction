local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local inherit = terralib.require("utils.inheritance")
local Node = terralib.require("tex.nodes.node")


-- This node provides the coordinates that are fed as inputs
--    to every other node.
-- It's a little different in that it has no inputs itself, so
--    it doesn't use the standard Node metatype.
local CoordSourceNode = S.memoize(function(real, GPU)

	local CoordScalarType = Vec(real, 2, GPU)
	local CoordVectorType
	if GPU then
		CoordVectorType = CUDAImage(real, 2, GPU)
	else
		CoordVectorType = image.Image(real, 2, GPU)
	end

	local struct CoordSourceNode(S.Object)
	{
		scalarCoord: CoordScalarType,
		vectorCoords: &CoordVectorType
	}
	local ParentNodeType = Node(real, 2, GPU)
	inherit.dynamicExtend(ParentNodeType, CoordSourceNode)

	terra CoordSourceNode:__init()
		self:initmembers()
		ParentNodeType.__init(self, nil)	-- No need to store any intermediate outputs
	end

	terra CoordSourceNode:setScalarCoord(x: real, y: real) : {}
		self.scalarCoord(0) = x
		self.scalarCoord(1) = y
	end

	terra CoordSourceNode:setScalarCoord(coord: CoordScalarType) : {}
		self:setScalarCoord(coord(0), coord(1))
	end

	terra CoordSourceNode:setVectorCoords(coords: &CoordVectorType)
		self.vectorCoords = coords
	end

	terra CoordSourceNode:evalScalarImpl() : CoordScalarType
		return self.scalarCoord
	end
	inherit.virtual(CoordSourceNode, "evalScalarImpl")

	terra CoordSourceNode:evalVectorImpl() : &CoordVectorType
		return self.vectorCoords
	end
	inherit.virtual(CoordSourceNode, "evalVectorImpl")

	return CoordSourceNode

end)


return CoordSourceNode




