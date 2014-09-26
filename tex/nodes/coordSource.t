local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local Registers = terralib.require("tex.registers")
local inherit = terralib.require("utils.inheritance")
local Node = terralib.require("tex.nodes.node").Node
local custd = terralib.require("utils.cuda.custd")


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
		vectorCoordXres: uint,
		vectorCoordYres: uint,
		vectorCoordXlo: real,
		vectorCoordXhi: real,
		vectorCoordYlo: real,
		vectorCoordYhi: real,
	}
	CoordSourceNode.metamethods.__typename = function(self)
		local platform = GPU and "GPU" or "CPU"
		return string.format("CoordSourceNode(%s, %s)", real, platform)
	end
	local ParentNodeType = Node(real, 2, GPU)
	inherit.dynamicExtend(ParentNodeType, CoordSourceNode)

	terra CoordSourceNode:__init(registers: &Registers(real, GPU))
		ParentNodeType.__init(self, registers)
	end

	-- Not implementing the shallowDuplicate virtual method, because this node should
	--    never be duplicated.

	-- Also not implementing setCoordInputNode, because this node is the root of the
	--    coordinate transformation graph. It is the root of all inputs.

	terra CoordSourceNode:setScalarCoord(x: real, y: real) : {}
		self.scalarCoord(0) = x
		self.scalarCoord(1) = y
	end

	terra CoordSourceNode:setScalarCoord(coord: CoordScalarType) : {}
		self:setScalarCoord(coord(0), coord(1))
	end

	terra CoordSourceNode:setVectorCoordRange(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
		self.vectorCoordXres = xres
		self.vectorCoordYres = yres
		self.vectorCoordXlo = xlo
		self.vectorCoordXhi = xhi
		self.vectorCoordYlo = ylo
		self.vectorCoordYhi = yhi
	end

	terra CoordSourceNode:evalScalarImpl() : CoordScalarType
		return self.scalarCoord
	end
	inherit.virtual(CoordSourceNode, "evalScalarImpl")

	terra CoordSourceNode:evalVectorImpl() : &CoordVectorType
		-- The result of this function is cached, so it's ok to do all this computation here.
		var xres = self.vectorCoordXres
		var yres = self.vectorCoordYres
		var xlo = self.vectorCoordXlo 
		var xhi = self.vectorCoordXhi 
		var ylo = self.vectorCoordYlo 
		var yhi = self.vectorCoordYhi
		var coords = self.imagePool:fetch(xres, yres)
		escape
			-- CPU: Sequential loop
			if not GPU then emit quote
				var xrange = xhi - xlo
				var yrange = yhi - ylo
				var xdelta = xrange / xres
				var ydelta = yrange / yres
				var yval = ylo
				for y=0,yres do
					var xval = xlo
					for x=0,xres do
						coords(x,y)(0) = xval
						coords(x,y)(1) = yval
						-- S.printf("%g, %g\n", xval, yval)
						xval = xval + xdelta
					end
					yval = yval + ydelta
				end
			end
			-- GPU: CUDA kernel
			-- This is kind of overkill, but it's actually convenient because coords is device-resident,
			--    so filling it in with a kernel is easier.
			else
				local lerp = macro(function(lo, hi, t) return `(1.0-t)*lo + t*hi end)
				local terra kernel(output: &Vec(real, 2, GPU), xres: uint, yres: uint, pitch: uint64,
								   xlo: real, xhi: real, ylo: real, yhi: real)
					var xi = custd.threadIdx.x()
					var yi = custd.blockIdx.x()
					var xt = xi/real(xres)
					var yt = yi/real(yres)
					var outptr = [&Vec(real, 2, GPU)]( [&uint8](output) + yi*pitch ) + xi
					(@outptr)(0) = lerp(xlo, xhi, xt)
					(@outptr)(1) = lerp(ylo, yhi, yt)
				end
				local K = terralib.cudacompile({kernel=kernel}, false)
				emit quote
					var cudaparams = terralib.CUDAParams { yres,1,1,  xres,1,1,  0, nil }
					K.kernel(&cudaparams, coords.data, xres, yres, coords.pitch, xlo, xhi, ylo, yhi)
				end
			end
		end
		return coords
	end
	inherit.virtual(CoordSourceNode, "evalVectorImpl")

	return CoordSourceNode

end)


return CoordSourceNode




