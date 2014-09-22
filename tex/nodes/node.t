local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local ImagePool = terralib.require("tex.imagePool")
local Registers = terralib.require("tex.registers")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local curt = terralib.require("utils.cuda.curt")
local custd = terralib.require("utils.cuda.custd")




-- Abstract base class for all texture nodes
local Node
Node = S.memoize(function(real, nchannels, GPU)

	-- IMPORTANT: All output channels of all nodes should always be in the range (0, 1)

	-- We support grayscale, color, and coordinate nodes.
	local isGrayscale = (nchannels == 1)
	local isCoordinate = (nchannels == 2)
	local isColor = (nchannels == 4)
	if not (isGrayscale or isCoordinate or isColor) then
		error(string.format("Node: nchannels was %s. Supported values are 1 (grayscale), 2 (coordinate), and 4 (color)",
			nchannels))
	end

	-- Define output types for scalar and vector execution
	local OutputScalarType = Vec(real, nchannels, GPU)
	local OutputVectorType
	if GPU then
		OutputVectorType = CUDAImage(real, nchannels)
	else
		OutputVectorType = image.Image(real, nchannels)
	end

	local struct NodeT(S.Object)
	{
		-- 'Registers' used to store vector interpreter intermediates
		imagePool: &ImagePool(real, nchannels, GPU),

		-- Bookkeeping for intermediate result caching
		scalarResult: OutputScalarType,
		vectorResult: &OutputVectorType,
		nOutputs: uint,
		nOutputsRemaining: uint
	}

	NodeT.OutputScalarType = OutputScalarType
	NodeT.OutputVectorType = OutputVectorType

	terra NodeT:__init(registers: &Registers(real, GPU))
		self:initmembers()
		S.assert(registers ~= nil)
		escape
			if isGrayscale then
				emit quote self.imagePool = &registers.grayscaleRegisters end
			elseif isCoordinate then
				emit quote self.imagePool = &registers.coordinateRegisters end
			elseif isColor then
				emit quote self.imagePool = &registers.colorRegisters end
			end
		end
		self.nOutputs = 0
		self.nOutputsRemaining = 0
	end

	-- Destructor does nothing, but it's virtual so that if subclass
	--    destructors need to do something, they can.
	terra NodeT:__destruct() : {} end
	inherit.virtual(NodeT, "__destruct")

	-- Scalar interpretation is only possible under CPU execution
	if not GPU then
		inherit.purevirtual(NodeT, "evalScalarImpl", {}->OutputScalarType)
		terra NodeT:evalScalar()
			-- If this is the first time we're evaluating this node on a run of the
			--    program (i.e. no outputs have yet requested this result), then
			--    compute and cache the result
			if self.nOutputsRemaining == self.nOutputs then
				self.scalarResult = self:evalScalarImpl()
			end
			-- Decrement the output count. If we're at zero (i.e. all outputs have been
			--    accounted for), then reset the remaining output count to prepare for
			--    the next run of the program.
			self.nOutputsRemaining = self.nOutputsRemaining - 1
			if self.nOutputsRemaining == 0 then
				self.nOutputsRemaining = self.nOutputs
			end
			return self.scalarResult
		end
	end

	inherit.purevirtual(NodeT, "evalVectorImpl", {}->&OutputVectorType)
	terra NodeT:evalVector()
		if self.nOutputsRemaining == self.nOutputs then
			self.vectorResult = self:evalVectorImpl()
		end
		self.nOutputsRemaining = self.nOutputsRemaining - 1
		return self.vectorResult
	end

	-- Release a vector output previously computed by this node
	terra NodeT:releaseVectorOutput()
		-- Only release if all outputs are finished with this result
		if self.nOutputsRemaining == 0 then
			self.nOutputsRemaining = self.nOutputs
			self.imagePool:release(self.vectorResult)
		end
	end

	-- Register that this node has one more additional output
	terra NodeT:incrementOutputCount()
		self.nOutputs = self.nOutputs + 1
		self.nOutputsRemaining = self.nOutputs
	end

	-- Register that this node has one fewer output
	terra NodeT:decrementOutputCount()
		self.nOutputs = self.nOutputs - 1
		self.nOutputsRemaining = self.nOutputs
	end



	---------------------------------------------------------
	-- Standard Node metatype function + related utilities --
	---------------------------------------------------------

	-- Add the coordinate input member to a node class
	local function addCoordinateInput(nodeClass)
		-- Member
		nodeClass.entries:insert({field="inputCoordNode", type=&Node(real, 2, GPU)})
		-- Getter
		nodeClass.methods.getInputCoordNode = terra(self: &nodeClass)
			return self.inputCoordNode
		end
		-- Setter
		nodeClass.methods.setInputCoordNode = terra(self: &nodeClass, coord: &Node(real, 2, GPU))
			if self.inputCoordNode ~= nil then
				self.inputCoordNode:decrementOutputCount()
			end
			self.inputCoordNode = coord
			coord:incrementOutputCount()
		end
	end

	-- Many texture nodes have input nodes; this utility allows easy creation of members for those
	--    inputs, as well as getters/setters
	function NodeT.defineInputs(nodeClass, numchannelsList)
		-- Throw an error if the class hasn't had coordinate input added to it; this must be added before
		--    any other inputs
		if not nodeClass.methods.setCoordInputNode then
			error(string.format("%s.defineInputs called on node class %s which is not using the standard metatype.",
				tostring(NodeT)))
		end
		for i,nchannels in ipairs(numchannelsList) do
			local typ = &Node(real, nchannels, GPU)
			-- Member
			nodeClass.entries:insert({field=string.format("inputNode%d",i), type=typ})
			-- Getter
			local getter = terra(self: &nodeClass)
				return self.[string.format("inputNode%d",i)]
			end
			getter:setinlined(true)
			nodeClass.methods[string.format("getInputNode%d",i)] = getter
			-- Setter
			local setter = terra(self: &nodeClass, input: typ)

				if self.[string.format("inputNode%d",i)] ~= nil then
					self.[string.format("inputNode%d",i)]:decrementOutputCount()
				end
				self.[string.format("inputNode%d",i)] = input
				input:incrementOutputCount()
			end
			setter:setinlined(true)
			nodeClass.methods[string.format("setInputNode%d",i)] = setter
		end
	end

	local function stringStartsWith(str, prefix)
	   return string.sub(str, 1, string.len(prefix)) == prefix
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
				if e.field == entry.field then
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
		local inputSyms = inputEntries:map(function(e) return symbol(e.type.type.OutputScalarType) end)
		local paramEntries = getParamEntries(nodeClass)
		-- Function body has to be wrapped in a macro to defer specialization until after class
		--    author has defined the 'eval' method
		local genFnBody = macro(function(self, ...)
			local inputs = {...}
			assert(nodeClass:getmethod("eval"),
				string.format("Texture node type %s must have an eval method", tostring(nodeClass)))
			local paramExps = paramEntries:map(function(e) return `self.[e.field] end)
			return `nodeClass.eval([inputs], [paramExps])
		end)
		nodeClass.methods.evalSelf = terra(self: &nodeClass, [inputSyms])
			return genFnBody(self, [inputSyms])
		end
	end

	-- Generate code for the virtual 'evalScalar' method
	local genEvalScalar = macro(function(self)
		local nodeClass = self:gettype().type
		-- evalScalar all of the inputs, then pass the results to eval.
		local inputs = getInputEntries(nodeClass)
		local inputResults = inputs:map(function(e) return `self.[e.field]:evalScalar() end)
		return ensureVecEval(`self:evalSelf([inputResults]), nodeClass)
	end)

	-- Generate code for the virtual 'evalVector' method
	local genEvalVector
	if not GPU then
		genEvalVector = macro(function(self)
			local nodeClass = self:gettype().type
			-- Fetch an image to use for our output.
			-- evalVector all of the inputs, then iterate over the results, calling eval.
			-- Release the images used for input results.
			local inputs = getInputEntries(nodeClass)
			local inputResults = inputs:map(function(e) return `self.[e.field]:evalVector() end)
			local inputTemps = inputs:map(function(e) return symbol(&e.type.type.OutputVectorType) end)
			local inputTempsAssign = quote var [inputTemps] = [inputResults] end
			local function inputTempsXY(x,y)
				return inputTemps:map(function(img) return `img(x,y) end)
			end
			local releaseInputTemps = inputs:map(function(e) return `self.[e.field]:releaseVectorOutput() end)
			return quote
				[inputTempsAssign]
				var xres = [inputTemps[1]].width
				var yres = [inputTemps[1]].height
				var outimg = self.imagePool:fetch(xres, yres)
				for y=0,yres do
					for x=0,xres do
						outimg(x,y) = [ensureVecEval(`self:evalSelf([inputTempsXY(x, y)]), nodeClass)]
					end
				end
				[releaseInputTemps]
			in
				outimg
			end
		end)
	else
		genEvalVector = macro(function(self)
			local nodeClass = self:gettype().type
			assert(nodeClass:getmethod("eval"),
				string.format("Texture node type %s must have an eval method", tostring(nodeClass)))
			-- We generate and compile a CUDA kernel, then return a quote that calls that kernel.
			local inputs = getInputEntries(nodeClass)
			local inputSyms = inputs:map(function(e) return symbol(&e.type.type.OutputScalarType) end)
			local inputPitchSyms = inputSyms:map(function(e) return symbol(uint64) end)
			local params = getParamEntries(nodeClass)
			local paramSyms = params:map(function(e) return symbol(e.type) end)
			local function inputTempsXY(xi, yi)
				local stmts = terralib.newlist()
				for i,s in ipairs(inputSyms) do
					local T = s.type.type
					stmts:insert( `@( [&T] ( [&uint8](s) + yi*[inputPitchSyms[i]] ) + xi ) )
				end
				return stmts
			end
			local terra kernel(output: &OutputScalarType, outpitch: uint64, [inputSyms], [inputPitchSyms], [paramSyms])
				var xi = custd.threadIdx.x()
				var yi = custd.blockIdx.x()
				var outptr = [&OutputScalarType]( [&uint8](output) + yi*outpitch ) + xi
				@outptr = [ensureVecEval(`nodeClass.eval([inputTempsXY(xi, yi)], [paramSyms]), nodeClass)]
			end
			local K = terralib.cudacompile({kernel = kernel}, false)
			local inputResults = inputs:map(function(e) return `self.[e.field]:evalVector() end)
			local inputTemps = inputs:map(function(e) return symbol(&e.type.type.OutputVectorType) end)
			local inputTempsAssign = quote var [inputTemps] = [inputResults] end
			local inputTempsData = inputTemps:map(function(img) return `img.data end)
			local inputTempsPitch = inputTemps:map(function(img) return `img.pitch end)
			local paramExps = params:map(function(e) return `self.[e.field] end)
			local releaseInputTemps = inputs:map(function(e) return `self.[e.field]:releaseVectorOutput() end)
			return quote
				[inputTempsAssign]
				var xres = [inputTemps[1]].width
				var yres = [inputTemps[1]].height
				var outimg = self.imagePool:fetch(xres, yres)
				-- Don't need an assert on xres, yres here b/c Program:interpretVector already has one
				-- TODO: If we implement caching and move that logic out of Program, we might need to reinstate the assert here.
				var cudaparams = terralib.CUDAParams { yres,1,1,  xres,1,1,  0, nil }
				K.kernel(&cudaparams, outimg.data, outimg.pitch, [inputTempsData], [inputTempsPitch], [paramExps])
				[releaseInputTemps]
			in
				outimg
			end
		end)
	end

	-- Initializes all of an object's input node pointers to nil. Subclasses must call this in their __init method.
	NodeT.methods.initInputs = macro(function(self)
		local nodeClass = self:gettype().type
		local inputs = getInputEntries(nodeClass)
		return quote
			escape
				for _,e in ipairs(inputs) do
					emit quote self.[e.field] = nil end
				end
			end
		end
	end)

	function NodeT.Metatype(nodeClass)

		addCoordinateInput(nodeClass)

		addEvalSelf(nodeClass)

		if not GPU then
			terra nodeClass:evalScalarImpl() : OutputScalarType
				return genEvalScalar(self)
			end
			inherit.virtual(nodeClass, "evalScalarImpl")
		end

		terra nodeClass:evalVectorImpl() : &OutputVectorType
			return genEvalVector(self)
		end
		inherit.virtual(nodeClass, "evalVectorImpl")

	end


	return NodeT

end)




return Node





