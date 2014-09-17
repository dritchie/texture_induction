local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local ImagePool = terralib.require("tex.imagePool")
local curt = terralib.includec("cuda_runtime.h")


-- Some random utilities that this file happens to need

local function stringStartsWith(str, prefix)
   return string.sub(str, 1, string.len(prefix)) == prefix
end

local lerp = macro(function(lo, hi, t)
	return (1.0-t)*lo + t*hi
end)

local terra getCUDADeviceProps()
	var devid : int
	curt.cudaGetDevice(&devid)
	var props : curt.cudaDeviceProp
	curt.cudaGetDeviceProperties(&props, devid)
	return props
end
local cudaDeviceProps = getCUDADeviceProps()



-- Abstract base class for all texture nodes
local Node
Node = S.memoize(function(real, nchannels, GPU)

	-- IMPORTANT: All output channels of all nodes should always be in the range (0, 1)

	local Image
	if GPU then
		Image = CUDAImage(real, nchannels)
	else
		Image = image.Image(real, nchannels)
	end

	local OutputType = Vec(real, nchannels, GPU)

	local struct NodeT(S.Object)
	{
		imagePool: &ImagePool(real, nchannels, GPU)
	}
	NodeT.OutputType = OutputType
	NodeT.ImageType = Image

	terra NodeT:__init(impool: &ImagePool(real, nchannels, GPU))
		self.imagePool = impool
	end

	-- Destructor does nothing, but it's virtual so that if subclass
	--    destructors need to do something, they can.
	terra NodeT:__destruct() : {} end
	inherit.virtual(NodeT, "__destruct")

	-- (Pointwise interpretation is only possible under CPU execution)
	if not GPU then
		-- Evaluate the texture function represented by this NodeT at the point (x,y)
		inherit.purevirtual(NodeT, "evalPoint", {real,real}->OutputType)

		-- Generates a texture image by intepreting the entire program graph rooted at this node for each pixel in the image.
		terra NodeT:interpretPixelwise(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
			var outimg = self.imagePool:fetch(xres, yres)
			var xrange = xhi - xlo
			var yrange = yhi - ylo
			var xdelta = xrange / xres
			var ydelta = yrange / yres
			var yval = ylo
			for y=0,yres do
				var xval = xlo
				for x=0,xres do
					outimg(x,y) = self:evalPoint(xval, yval)
					xval = xval + xdelta
				end
				yval = yval + ydelta
			end
			return outimg
		end
	end

	-- Evaluate the texture function over an entire image
	inherit.purevirtual(NodeT, "evalImage", {uint,uint,real,real,real,real}->&Image)

	-- Generates a texture image by interpreting the program graph rooted at this node one node at a time, generating
	--    an entire image at each stage.
	terra NodeT:interpretNodewise(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
		escape
			if GPU then
				-- (For now) Can't generate an image at a resolution that exceeds the device's maximum block dimensions
				-- TODO: Could switch to parallelizing over grids instead of blocks, but I don't think that's as efficient...
				emit quote S.assert(xres <= cudaDeviceProps.maxThreadsDim[0] and yres <= cudaDeviceProps.maxThreadsDim[1]) end
			end
		end
		return self:evalImage(xres, yres, xlo, xhi, ylo, yhi)
	end


	---------------------------------------------------------
	-- Standard Node metatype function + related utilities --
	---------------------------------------------------------

	-- Many texture nodes have input nodes; this utility allows easy creation of members for those
	--    inputs, as well as getters/setters
	function NodeT.defineInputs(nodeClass, numchannelsList)
		for i,nchannels in ipairs(numchannelsList) do
			local typ = &Node(real, nchannels, GPU)
			-- Member
			nodeClass.entries:insert({field=string.format("input%d",i), type=typ})
			-- Getter
			local getter = terra(self: &nodeClass)
				return self.[string.format("input%d",i)]
			end
			getter:setinlined(true)
			nodeClass.methods[string.format("getInput%d",i)] = getter
			-- Setter
			local setter = terra(self: &nodeClass, input: typ)
				self.[string.format("input%d",i)] = input
			end
			setter:setinlined(true)
			nodeClass.methods[string.format("setInput%d",i)] = setter
		end
	end

	-- Fetch all entries of a node struct that correspond to node inputs
	local function getInputEntries(nodeClass)
		local lst = terralib.newlist()
		for _,entry in ipairs(nodeClass.entries) do
			if stringStartsWith(entry.field, "input") then
				lst:insert(entry)
			end
		end
		return lst
	end

	-- Fetch all entries of a node struct that correspond to node parameters
	-- (These are all entries of the struct, minus the inputs and anything defined
	--    on the Node base class)
	local function getParamEntries(nodeClass)
		local function isDefinedOnBaseClass(entry)
			for _,e in ipairs(NodeT.entries) do
				if e.field == entry.field and e.type == entry.type then
					return true
				end
			end
			return false
		end
		local function isInputEntry(entry)
			return stringStartsWith(entry.field, "input")
		end
		local lst = terralib.newlist()
		for _,e in ipairs(nodeClass.entries) do
			if not isInputEntry(e) and not isDefinedOnBaseClass(e) then
				lst:insert(e)
			end
		end
		return lst
	end

	-- If the number of channels is 1, then 'eval' returns real and we need
	--    to convert that to a Vec(real, 1) to be consistent with downstream code.
	local function ensureVecEval(expr, nodeClass)
		if nchannels == 1 then
			local VecT = Vec(real, 1, GPU)
			return quote
				var v : VecT
				v:init(expr)
			in
				v
			end
		else
			return expr
		end
	end

	-- Add the 'evalSelf' method (the version of eval that uses parameters stored on 'self'
	--    instead of passed in explicitly)
	local function addEvalSelf(nodeClass)
		local inputEntries = getInputEntries(nodeClass)
		local inputSyms = inputEntries:map(function(e) return symbol(e.type) end)
		local paramEntries = getParamEntries(nodeClass)
		-- Function body has to be wrapped in a macro to defer specialization until after class
		--    author has defined the 'eval' method
		local genFnBody = macro(function(self, x, y, ...)
			local inputs = {...}
			assert(nodeClass:getmethod("eval"),
				string.format("Texture node type %s must have an eval method", tostring(nodeClass)))
			local paramExps = paramEntries:map(function(e) return `self.[e.field] end)
			return `nodeClass.eval(x, y, [inputs], [paramExps])
		end)
		nodeClass.methods.evalSelf = terra(self: &nodeClass, x: real, y: real, [inputSyms])
			return genFnBody(self, x, y, [inputSyms])
		end
	end

	-- Generate code for the virtual 'evalPoint' method
	local genEvalPoint = macro(function(self, x, y)
		local nodeClass = self:gettype().type
		-- evalPoint all of the inputs, then pass the results to eval.
		local inputs = getInputEntries(nodeClass)
		local inputResults = inputs:map(function(e) return `self.[e.field]:evalPoint(x,y) end)
		return ensureVecEval(`self:evalSelf(x, y, [inputResults]), nodeClass)
	end)

	-- Generate code for the virtual 'evalImage' method
	-- CPU version iterates over every pixel
	local genEvalImage
	if not GPU then
		genEvalImage = macro(function(self, xres, yres, xlo, xhi, ylo, yhi)
			local nodeClass = self:gettype().type
			-- Fetch an image to use for our output.
			-- evalImage all of the inputs, then iterate over the results, calling eval.
			-- Release the images used for input results.
			local inputs = getInputEntries(nodeClass)
			local inputResults = inputs:map(function(e) return `self.[e.field]:evalImage(xres,yres,xlo,ylo,yhi) end)
			local inputTemps = inputs:map(function(e) return symbol(&e.type.type.ImageType) end)
			local inputTempsAssign = #inputTemps > 0 and
				quote var [inputTemps] = [inputResults] end
			or
				quote end
			local function inputTempsXY(x,y)
				return inputTemps:map(function(img) return `img(x,y) end)
			end
			local freeInputResults = inputTemps:map(function(img) return `self.imagePool:release(img) end)
			return quote
				var outimg = self.imagePool:fetch(xres, yres)
				[inputTempsAssign]
				var xrange = xhi - xlo
				var yrange = yhi - ylo
				var xdelta = xrange / xres
				var ydelta = yrange / yres
				var yval = ylo
				for y=0,yres do
					var xval = xlo
					for x=0,xres do
						outimg(x,y) = [ensureVecEval(`self:evalSelf(xval, yval, [inputTempsXY(xval, yval)]), nodeClass)]
						xval = xval + xdelta
					end
					yval = yval + ydelta
				end
				[freeInputResults]
			in
				outimg
			end
		end)
	-- GPU version defines a CUDA kernel that does the above in parallel
	else
		genEvalImage = macro(function(self, xres, yres, xlo, xhi, ylo, yhi)
			local nodeClass = self:gettype().type
			-- We generate and compile a CUDA kernel, then return a quote that calls that kernel.
			local inputs = getInputEntries(nodeClass)
			local inputSyms = inputs:map(function(e) return symbol(&e.type.type.OutputType) end)
			local params = getParamEntries(nodeClass)
			local paramSyms = params:map(function(e) return symbol(e.type) end)
			local function inputTempsXY(xi, yi, pitch)
				return inputSyms:map(function(s)
					local T = s.type.type
					return `@( [&T] ( [&uint8](s) + yi*pitch ) + xi )
				end)
			end
			local terra kernel(output: &OutputType, width: uint, height: uint, pitch: uint, xlo: real, xhi: real, ylo: real, yhi: real,
							  [inputSyms], [paramSyms])
				var xi = cudalib.nvvm_read_ptx_sreg_tid_x
				var yi = cudalib.nvvm_read_ptx_sreg_tid_y
				var xt = xi / real(width)
				var yt = yi / real(height)
				var x = lerp(xlo, xhi, xt)
				var y = lerp(ylo, yhi, yt)
				var outptr = [&OutputType]( [&uint8](output) + yi*pitch ) + xi
				@outptr = nodeClass.eval(x, y, [inputTempsXY(xi, yi, pitch)], [paramSyms])
			end
			local K = terralib.cudacompile({kernel = kernel}, false)
			local inputResults = inputs:map(function(e) return `self.[e.field]:evalImage(xres,yres,xlo,ylo,yhi) end)
			local inputTemps = inputs:map(function(e) return symbol(&e.type.type.ImageType) end)
			local inputTempsAssign = #inputTemps > 0 and
				quote var [inputTemps] = [inputResults] end
			or
				quote end
			local inputTempsData = inputTemps:map(function(img) return `img.data end)
			local paramExps = params:map(function(e) return `self.[e.field] end)
			local freeInputResults = inputTemps:map(function(img) return `self.imagePool:release(img) end)
			return quote
				var outimg = self.imagePool:fetch(xres, yres)
				[inputTempsAssign]
				var cudaparams = terralib.CUDAParams { 1,1,1,  xres,yres,1,  0, nil }
				K.kernel(&cudaparams, outimg.data, xres, yres, outimg.pitch, xlo, xhi, ylo, yhi, [inputTempsData], [paramExps])
				[freeInputResults]
			in
				outimg
			end
		end)
	end

	function NodeT.Metatype(nodeClass)

		addEvalSelf(nodeClass)

		if not GPU then
			terra nodeClass:evalPoint(x: real, y: real) : OutputType
				return genEvalPoint(self, x,y)
			end
			inherit.virtual(nodeClass, "evalPoint")
		end

		terra nodeClass:evalImage(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
								  : &Image
			return genEvalImage(self, xres,yres,xlo,xhi,ylo,yhi)
		end
		inherit.virtual(nodeClass, "evalImage")

	end


	return NodeT

end)




return Node





