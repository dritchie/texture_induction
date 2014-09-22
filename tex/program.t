local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local Node = terralib.require("tex.nodes.node").Node
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local Registers = terralib.require("tex.registers")
local CoordSourceNode = terralib.require("tex.nodes.coordSource")
local curt = terralib.require("utils.cuda.curt")


-- Code later on makes use of some CUDA device properties, so we'll
--    pre-fetch that data.
-- TODO: Refactor to move this elsewhere / make more general? 
local terra getCUDADeviceProps()
	var devid : int
	curt.cudaGetDevice(&devid)
	var props : curt.cudaDeviceProp
	curt.cudaGetDeviceProperties(&props, devid)
	return props
end
local cudaDeviceProps = getCUDADeviceProps()



-- A procedural texture program. Contains a DAG of Nodes, one of which is the
--    output color, and another one of which is the input coordinate(s).
-- A program can be executed in one of several different ways:
--    * Scalar interpreter: Walks the graph and evaluates each node for a single
--         (x,y) coordinate
--    * Vector interpreter: Walks the graphs and evalutes each node for a block
--         of (x,y) coordinates (an entire image)
--    * Compiler: Compiles the program so that it can be executed with no interpreter
--         overhead.
-- A program can be executed on either the CPU or GPU. The scalar interpreter is
--    only available for the CPU.
local Program = S.memoize(function(real, nOutChannels, GPU)

	-- A program can have grayscale or color output
	local isGrayscale = (nOutChannels == 1)
	local isColor = (nOutChannels == 4)
	if not (isGrayscale or isColor) then
		error(string.format("Program: nOutChannels was %s. Supported values are 1 (grayscale)  and 4 (color)",
			nOutChannels))
	end

	local Image
	if GPU then
		Image = CUDAImage(real, nOutChannels)
	else
		Image = image.Image(real, nOutChannels)
	end

	local struct Program(S.Object)
	{
		-- The input and output nodes
		inputCoordNode: CoordSourceNode(real, GPU),
		outputNode: &Node(real, nOutChannels, GPU)
	}

	terra Program:__init(registers: &Registers(real, GPU))
		self.inputCoordNode:init(registers)
		self.outputNode = nil
	end

	-- Retrieve a pointer to the coord source node for this program.
	terra Program:getInputCoordNode() return &self.inputCoordNode end

	-- Output node must be set before the program can be executed.
	terra Program:setOuputNode(outnode: &Node(real, nOutChannels, GPU))
		if self.outputNode ~= nil then
			self.outputNode:decrementOutputCount()
		end
		self.outputNode = outnode
		outnode:incrementOutputCount()
	end

	-- The scalar interpreter
	if not GPU then
		terra Program:interpretScalar(outimg: &Image, xlo: real, xhi: real, ylo: real, yhi: real)
			var xrange = xhi - xlo
			var yrange = yhi - ylo
			var xdelta = xrange / outimg.width
			var ydelta = yrange / outimg.height
			var yval = ylo
			for y=0,outimg.height do
				var xval = xlo
				for x=0,outimg.width do
					self.inputCoordNode:setScalarCoord(xval, yval)
					outimg(x,y) = self.outputNode:evalScalar()
					xval = xval + xdelta
				end
				yval = yval + ydelta
			end
		end
	end

	-- The vector interpreter
	terra Program:interpretVector(outimg: &Image, xlo: real, xhi: real, ylo: real, yhi: real)
		var xres = outimg.width
		var yres = outimg.height
		escape
			if GPU then
				emit quote
					-- We compute one scanline per thread block, so yres must be less than the maximum block dimension, and
					--    xres must be less than the maximum thread dimension.
					S.assert(yres <= cudaDeviceProps.maxGridSize[0] and xres <= cudaDeviceProps.maxThreadsDim[0])
				end
			end
		end
		self.inputCoordNode:setVectorCoordRange(xres, yres, xlo, xhi, ylo, yhi)
		var outtmp = self.outputNode:evalVector()
		outimg:memcpy(outtmp)
		self.outputNode:releaseVectorOutput()
	end


	-- TODO: Implement the compiler


	return Program

end)


return Program



