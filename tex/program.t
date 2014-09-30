local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local node = terralib.require("tex.functions.node")
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local Registers = terralib.require("tex.registers")
local CoordSourceNode = terralib.require("tex.functions.coordSource")
local Function = terralib.require("tex.functions.function")
local HashMap = terralib.require("qs.lib.hashmap")
local curt = terralib.require("utils.cuda.curt")
local custd = terralib.require("utils.cuda.custd")
local cudaWrapKernel = terralib.require("utils.cuda.cukernelwrap")

local Node = node.Node


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

	local Image
	if GPU then
		Image = CUDAImage(real, nOutChannels)
	else
		Image = image.Image(real, nOutChannels)
	end

	local struct Program(S.Object)
	{
		rootFn: &Function(real, nOutChannels, GPU),
		inputCoordNode: &CoordSourceNode(real, GPU),
		outputNode: &Node(real, nOutChannels, GPU)
	}

	-- Assumes ownership of rootFn
	terra Program:__init(registers: &Registers(real, GPU), rootFn: &Function(real, nOutChannels, GPU))
		self.rootFn = rootFn
		self.inputCoordNode = [CoordSourceNode(real, GPU)].alloc():init(registers)
		self.outputNode = rootFn:expand(self.inputCoordNode)
		self.outputNode:incrementOutputCount()
	end

	terra Program:__destruct()
		self.rootFn:delete()
		-- This will in turn recursively delete the inputCoordNode
		self.outputNode:delete()
	end

	terra Program:treePrintPretty()
		self.rootFn:treePrintPretty(0)
	end

	terra Program:ssaPrintPretty()
		var addrToId = [HashMap(&opaque, uint)].salloc():init()
		self.rootFn:ssaPrintPretty(addrToId)
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


	-- The compiler
	-- NOTE: This is a Lua function. This isn't meant to be invoked from Terra code.
	Program.methods.compile = function(program)
		-- First, compile a scalar (pointwise) program
		local selfsymbol = symbol(&Program)
		local scalarfn, paramExtractExprs = Program.methods.compileScalar(program, selfsymbol)
		-- Then generate a wrapper function that invokes this function on every pixel
		--    of an output image.
		return terra([selfsymbol], outimg: &Image, xlo: real, xhi: real, ylo: real, yhi: real)
			var xres = outimg.width
			var yres = outimg.height
			escape
				if not GPU then
					emit quote
						var coord = @[Vec(real, 2, false)].salloc():init()
						var xrange = xhi - xlo
						var yrange = yhi - ylo
						var xdelta = xrange / outimg.width
						var ydelta = yrange / outimg.height
						var yval = ylo
						for y=0,outimg.height do
							coord(1) = yval
							var xval = xlo
							for x=0,outimg.width do
								coord(0) = xval
								outimg(x,y) = scalarfn(coord, [paramExtractExprs])
								xval = xval + xdelta
							end
							yval = yval + ydelta
						end
					end
				else
					-- We need to pass in all the scalarfn parameters, minus the first
					--    parameter ('coord'), to the CUDA kernel.
					local paramSymbols = terralib.newlist()
					local scalarfnType = scalarfn:gettype()
					for i=2,#scalarfnType.parameters do
						paramSymbols:insert(symbol(scalarfnType.parameters[i]))
					end
					local lerp = macro(function(lo, hi, t) return `(1.0-t)*lo + t*hi end)
					local terra kernel(output: &Vec(real, nOutChannels, true), xres: uint, yres: uint, pitch: uint64,
									   xlo: real, xhi: real, ylo: real, yhi: real, [paramSymbols])
						var xi = custd.threadIdx.x()
						var yi = custd.blockIdx.x()
						var xt = xi/real(xres)
						var yt = yi/real(yres)
						var x = lerp(xlo, xhi, xt)
						var y = lerp(ylo, yhi, yt)
						var coord = @[Vec(real, 2, true)].salloc():init(x, y)
						var outptr = [&Vec(real, nOutChannels, true)]( [&uint8](output) + yi*pitch ) + xi
						@outptr = scalarfn(coord, [paramSymbols])
					end
					local wkernel = cudaWrapKernel(kernel)
					emit quote
						S.assert(yres <= cudaDeviceProps.maxGridSize[0] and xres <= cudaDeviceProps.maxThreadsDim[0])
						var cudaparams = terralib.CUDAParams { yres,1,1,  xres,1,1,  0, nil }
						wkernel(&cudaparams, outimg.data, outimg.width, outimg.height, outimg.pitch, xlo, xhi, ylo, yhi, [paramExtractExprs])
					end
				end
			end
		end
	end

	-- Compile the pointwise program that is needed by 'compile' above
	-- Returns the program, as well as a list of expressions (defined in terms of
	--    selfsymbol) that extract from a Program the parameters required to evaluate it.
	Program.methods.compileScalar = function(program, selfsymbol)
		-- SSA assigment statements that will make up the function body
		local bodyStmts = terralib.newlist()
		-- Symbols for each program parameter that will go into the function formal param list.
		local paramSyms = terralib.newlist()
		-- Expressions that extract parameters from the program object
		local paramExtractExprs = terralib.newlist()
		-- A map from Node pointer to the symbol that stores the intermediate result of evaluating
		--    that Node. Used to avoid reduandant recomputation of shared inputs.
		local immCache = {}
		-- An expression referring to the currently-visted node, in terms of the program object
		--    (a big chain of casts and dereferences). Useful for building up paramExtractExprs.
		local selfExpr = `selfsymbol.outputNode

		-- We seed the immCache with a symbol that refers to the input (x,y) coordinate.
		local inputNodeID = tostring(terralib.cast(uint64, program.inputCoordNode))
		local coordSym = symbol(Vec(real, 2, GPU))
		immCache[inputNodeID] = coordSym

		-- Then we recursively walk the program graph to fill in all the above tables
		local retvalSym = node.generateCode(program.outputNode, bodyStmts, paramSyms, paramExtractExprs, immCache, selfExpr)

		-- Finally, we generate the Terra function
		local scalarfn = terra([coordSym], [paramSyms])
			[bodyStmts]
			return retvalSym
		end

		return scalarfn, paramExtractExprs
	end


	return Program

end)


return Program



