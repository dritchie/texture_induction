local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local Node = terralib.require("tex.nodes.node")
local Registers = terralib.require("tex.registers")
local CoordSourceNode = terralib.require("tex.nodes.coordSource")
local curt = terralib.require("utils.cuda.curt")
local custd = terralib.require("utils.cuda.custd")


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

	local struct Program(S.Object)
	{
		registers: &Registers(real, GPU),

		-- The input and output nodes
		inputCoordNode: CoordSourceNode(real, GPU),
		outputNode: &Node(real, nOutChannels, GPU)
	}

	terra Program:__init(registers: &Registers(real, GPU))
		self:initmembers()
		self.outputNode = nil
		self.registers = registers
	end

	-- Macro that returns the register that corresponds to this program's output type
	Program.methods.outputRegisters = macro(function(self)
		if isGrayscale then
			return `self.registers.grayscaleRegisters
		elseif isColor then
			return `self.registers.colorRegisters
		end
	end)

	-- Retrieve a pointer to the coord source node for this program.
	terra Program:getInputCoordNode() return &self.inputCoordNode end

	-- Output node must be set before the program can be executed.
	terra Program:setOuputNode(outnode: &Node(real, nOutChannels, GPU))
		self.outputNode = outnode
	end

	-- The scalar interpreter
	if not GPU then
		terra Program:interpretScalar(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
			var outimg = self:outputRegisters():fetch(xres, yres)
			var xrange = xhi - xlo
			var yrange = yhi - ylo
			var xdelta = xrange / xres
			var ydelta = yrange / yres
			var yval = ylo
			for y=0,yres do
				var xval = xlo
				for x=0,xres do
					self.inputCoordNode:setScalarCoord(xval, yval)
					outimg(x,y) = self.outputNode:evalScalar()
					xval = xval + xdelta
				end
				yval = yval + ydelta
			end
			-- Caller is responsible for releasing the output image (e.g. via self:clearOutputRegisters).
			return outimg
		end
	end

	-- The vector interpreter
	terra Program:interpretVector(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
		var coords = self.registers.coordinateRegisters:fetch(xres, yres)
		-- Fill in coords
		-- TODO: If we had a caching system for intermediate ouputs (see the TODO in node.t), then
		--    this could be moved to coordSource.t without loss of efficiency.
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
						xval = xval + xdelta
					end
					yval = yval + ydelta
				end
			end
			-- GPU: CUDA kernel
			-- This is totally overkill, but it's actually convenient because coords is device-resident,
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
					-- We compute one scanline per thread block, so yres must be less than the maximum block dimension, and
					--    xres must be less than the maximum thread dimension.
					S.assert(yres <= cudaDeviceProps.maxGridSize[0] and xres <= cudaDeviceProps.maxThreadsDim[0])
					var cudaparams = terralib.CUDAParams { yres,1,1,  xres,1,1,  0, nil }
					K.kernel(&cudaparams, coords.data, xres, yres, coords.pitch, xlo, xhi, ylo, yhi)
				end
			end
		end
		self.inputCoordNode:setVectorCoords(coords)
		var outimg = self.outputNode:evalVector()
		self.registers.coordinateRegisters:release(coords)
		-- Caller is responsible for releasing the output image (e.g. via self:clearOutputRegisters).
		return outimg
	end

	-- The output of program when executed under interpretation is an image from the pool of output
	--    registers. This method clears those registers for future use.
	terra Program:clearOuputRegisters()
		self:outputRegisters():releaseAll()
	end


	-- TODO: Implement the compiler


	return Program

end)


return Program



