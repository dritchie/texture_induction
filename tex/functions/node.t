local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local ImagePool = terralib.require("tex.imagePool")
local Registers = terralib.require("tex.registers")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local curt = terralib.require("utils.cuda.curt")
local custd = terralib.require("utils.cuda.custd")
local cudaWrapKernel = terralib.require("utils.cuda.cukernelwrap")




-- Associating a unique identifier with every texture node class
local nodeTypeID = 1
local idToType = {}




-- Abstract base class for all texture nodes
local Node
Node = S.memoize(function(real, nchannels, GPU)
	assert(type(nchannels) == "number", "Node: nchannels must be a number")
	assert(type(GPU) == "boolean", "Node: GPU must be a boolean")

	-- IMPORTANT: All output channels of all nodes should always be in the range (0, 1)

	if nchannels < 1 or nchannels > 4 then
		error(string.format("Node: nchannels was %d. Supported values are 1, 2, 3, and 4",
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
		-- type ID (for Lua introspection)
		typeID: uint,

		-- 'Registers' used to store vector interpreter intermediates
		imagePool: &ImagePool(real, nchannels, GPU),

		-- Bookkeeping for intermediate result caching
		scalarResult: OutputScalarType,
		vectorResult: &OutputVectorType,
		nOutputs: uint,
		nOutputsRemaining: uint
	}
	NodeT.metamethods.__typename = function(self)
		local platform = GPU and "GPU" or "CPU"
		return string.format("Node(%s, %d, %s)", real, nchannels, platform)
	end

	NodeT.OutputScalarType = OutputScalarType
	NodeT.OutputVectorType = OutputVectorType

	terra NodeT:__init(registers: &Registers(real, GPU))
		self:initmembers()
		self.typeID = 0		-- Subclasses should set this to a positive value
		S.assert(registers ~= nil)
		self.imagePool = &registers.[string.format("vec%dRegisters",nchannels)]
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

	terra NodeT:getNumOutputs() return self.nOutputs end



	---------------------------------------------------------
	-- Standard Node metatype function + related utilities --
	---------------------------------------------------------

	local function defineInputs(nodeClass, numchannelsList)
		assert(numchannelsList ~= nil,
			"makeNodeFromFunc: number of channels for each input was not provided.")
		for i,nchannels in ipairs(numchannelsList) do
			local typ = &Node(real, nchannels, GPU)
			-- Member
			nodeClass.entries:insert({field=string.format("inputNode%d",i), type=typ})
			-- Getter
			local getter = terra(self: &nodeClass)
				return self.[string.format("inputNode%d",i)]
			end
			nodeClass.methods[string.format("getInputNode%d",i)] = getter
			-- Setter
			local setter = terra(self: &nodeClass, input: typ)

				if self.[string.format("inputNode%d",i)] ~= nil then
					self.[string.format("inputNode%d",i)]:decrementOutputCount()
				end
				self.[string.format("inputNode%d",i)] = input
				input:incrementOutputCount()
			end
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
	NodeT.getInputEntries = getInputEntries

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
	NodeT.getParamEntries = getParamEntries

	-- If the number of channels is 1 and 'eval' returns real, then we need
	--    to convert that to a Vec(real, 1) to be consistent with downstream code.
	local ensureVecEval = macro(function(exp)
		if exp:gettype() == real then
			return `[Vec(real, 1, GPU)].create(exp)
		else
			return exp
		end
	end)
	NodeT.ensureVecEval = ensureVecEval

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
		return `ensureVecEval(self:evalSelf([inputResults]))
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
						outimg(x,y) = ensureVecEval(self:evalSelf([inputTempsXY(x, y)]))
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
				@outptr = ensureVecEval(nodeClass.eval([inputTempsXY(xi, yi)], [paramSyms]))
			end
			local wkernel = cudaWrapKernel(kernel)
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
				var cudaparams = terralib.CUDAParams { yres,1,1,  xres,1,1,  0, nil }
				wkernel(&cudaparams, outimg.data, outimg.pitch, [inputTempsData], [inputTempsPitch], [paramExps])
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

	function NodeT.Metatype(nodeClass, inputNChannelsList)

		nodeClass.NodeTypeID = nodeTypeID
		idToType[nodeTypeID] = nodeClass
		nodeTypeID = nodeTypeID + 1

		defineInputs(nodeClass, inputNChannelsList)

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



-- Make a node subclass from an eval function template
-- The template should return the eval function, as well as a table containing the number
--    of channels for each of the node's inputs.
-- For a node with no inputs (beyond the obligatory coordinate), return an empty table.
local function makeNodeFromFunc(nodeName, fnTemplate)
	return S.memoize(function(...)
		local args = {...}
		local evalFn, inputNChannelsList = fnTemplate(...)
		evalFn:setinlined(true)
		local evalFnType = evalFn:gettype()
		assert(inputNChannelsList ~= nil,
			"makeNodeFromFunc: number of channels for each input was not provided.")

		-- Args are either
		--    * real, nchannels, GPU
		--    * real, GPU
		-- depending on whether the node supports different channel numbers
		local real = args[1]
		local GPU
		if #args == 3 then
			GPU = args[3]
		else
			GPU = args[2]
		end
		local nchannels
		if #args == 3 then
			nchannels = args[2]
		else
			local isGrayscale = (evalFnType.returntype == real)
			local isCoordinate = (evalFnType.returntype == Vec(real, 2, GPU))
			local isColor = (evalFnType.returntype == Vec(real, 4, GPU))
			nchannels = (isGrayscale and 1 or (isCoordinate and 2 or (isColor and 4)))
		end

		assert(type(nchannels) == "number", string.format("%s: nchannels must be a number", nodeName))
		assert(type(GPU) == "boolean", string.format("%s: GPU must be a boolean", nodeName))

		-- Set up struct
		local struct NodeClass(S.Object) {}
		NodeClass.metamethods.__typename = function(self)
			local platform = GPU and "GPU" or "CPU"
			if #args == 3 then
				return string.format("%s(%s, %d, %s)", nodeName, real, nchannels, platform)
			else
				return string.format("%s(%s, %s)", nodeName, real, platform)
			end
		end
		local ParentNodeType = Node(real, nchannels, GPU)
		inherit.dynamicExtend(ParentNodeType, NodeClass)
		NodeClass.ParentNodeType = ParentNodeType

		NodeClass.methods.eval = evalFn

		-- Add struct entries for each parameter. We figure out the parameters
		--    by inspecting the parameter list of the eval function. If the node has
		--    n inputs, then the first n eval parameters correspond to the these inputs.
		--    The remaining parameters are the parameters we're looking for
		local index = 1
		local paramSyms = terralib.newlist()
		for i=#inputNChannelsList+1,#evalFnType.parameters do
			local paramtype = evalFnType.parameters[i]
			NodeClass.entries:insert({field=string.format("param%d",index), type=paramtype})
			paramSyms:insert(symbol(paramtype))
			index = index + 1
		end

		-- Now that we have the parameters as struct entries, we can apply the metatype
		ParentNodeType.Metatype(NodeClass, inputNChannelsList)

		-- Constructor just stores the inputs and parameters (as well as the ever-present
		--    vector registers)
		local inputSyms = ParentNodeType.getInputEntries(NodeClass):map(function(e) return symbol(e.type) end)
		terra NodeClass:__init(registers: &Registers(real, GPU), [inputSyms], [paramSyms]) : {}
			ParentNodeType.__init(self, registers) -- *must* be called (to init self.imagePool)
			ParentNodeType.initInputs(self)		   -- *must* be called (to init input node pointers to nil)
			escape
				-- Store input nodes
				for i,is in ipairs(inputSyms) do
					emit quote self:[string.format("setInputNode%d",i)](is) end
				end
				-- Store params
				for i,ps in ipairs(paramSyms) do
					emit quote self.[string.format("param%d",i)] = ps end
				end
			end
			self.typeID = [NodeClass.NodeTypeID]   -- *must* be set for compiler code gen to work
		end

		-- Destructor goes through all the inputs and deletes them if they are not nil
		terra NodeClass:__destruct() : {}
			escape
				local inputs = ParentNodeType.getInputEntries(NodeClass)
				for _,e in ipairs(inputs) do
					emit quote
						if self.[e.field] ~= nil then
							self.[e.field]:delete()
							self.[e.field] = nil
						end
					end
				end
			end
		end
		inherit.virtual(NodeClass, "__destruct")

		return NodeClass

	end)
end


-- Recursively generate code for the function body of a compiled texture program.
local genCodeForClass
local function generateCode(node, bodyStmts, paramSyms, paramExtractExprs, immCache, selfExpr)
	-- Don't generate if we've already done so and cached the result
	local nodekey = tostring(terralib.cast(uint64, node))
	local retsym = immCache[nodekey]
	if not retsym then
		local nodeClass = idToType[node.typeID]
		assert(nodeClass, string.format("generateCode found node typeID %d that did not map to any Node type.", node.typeID))
		retsym = genCodeForClass(nodeClass, terralib.cast(&nodeClass, node), bodyStmts, paramSyms,
			paramExtractExprs, immCache, `[&nodeClass](selfExpr))
		immCache[nodekey] = retsym
	end
	return retsym
end

genCodeForClass = function(nodeClass, node, bodyStmts, paramSyms, paramExtractExprs, immCache, selfExpr)
	local ParentNodeType = nodeClass.ParentNodeType
	-- Recursively ensure code has been generated for inputs
	local myInputs = ParentNodeType.getInputEntries(nodeClass)
	local myInputSyms = myInputs:map(function(ie)
		local inNode = node[ie.field]
		local inSelfExpr = `selfExpr.[ie.field]
		return generateCode(inNode, bodyStmts, paramSyms, paramExtractExprs, immCache, inSelfExpr)
	end)
	-- Fill in params stuff
	local myParams = ParentNodeType.getParamEntries(nodeClass)
	local myParamSyms = terralib.newlist()
	for _,pe in ipairs(myParams) do
		local psym = symbol(pe.type)
		myParamSyms:insert(psym)
		paramSyms:insert(psym)
		paramExtractExprs:insert(`selfExpr.[pe.field])
	end
	-- Generate code for this node
	local retsym = symbol(ParentNodeType.OutputScalarType)
	bodyStmts:insert(quote
		var [retsym] = [ParentNodeType.ensureVecEval](nodeClass.eval([myInputSyms], [myParamSyms]))
	end)
	return retsym
end


return
{
	Node = Node,
	makeNodeFromFunc = makeNodeFromFunc,
	generateCode = generateCode
}





