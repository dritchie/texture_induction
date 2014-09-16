local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local ImagePool = terralib.require("tex.imagePool")


function stringStartsWith(str, prefix)
   return string.sub(str, 1, string.len(prefix)) == prefix
end



-- Abstract base class for all texture nodes
local Node
Node = S.memoize(function(real, nchannels, GPU)

	-- IMPORTANT: All output channels of all nodes should always be in the range (0, 1)

	local OutputType = Vec(real, nchannels, GPU)

	local struct NodeT(S.Object)
	{
		imagePool: &ImagePool(real, nchannels, GPU)
	}

	terra NodeT:__init(impool: &ImagePool(real, nchannels, GPU))
		self.imagePool = impool
	end

	-- Destructor does nothing, but it's virtual so that if subclass
	--    destructors need to do something, they can.
	terra NodeT:__destruct() : {} end
	inherit.virtual(NodeT, "__destruct")

	-- Evaluate the texture function represented by this NodeT at the point (x,y)
	inherit.purevirtual(NodeT, "evalPoint", {real,real}->OutputType)

	-- Generate a texture image from the graph rooted at this node, evaluating the entire graph at
	--    every pixel.
	terra NodeT:genTexturePointwise(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
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

	-- Evaluate the texture function over an entire image
	inherit.purevirtual(NodeT, "evalImage", {uint,uint,real,real,real,real}->&image.Image(real,nchannels))

	-- Generate a texture image from the graph rooted at this node, evaluating an entire image for
	--    each stage in the graph.
	terra NodeT:genTextureBlocked(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
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

	-- Generate code for the virtual 'evalPoint' method, given an 'eval' method
	local genEvalPoint = macro(function(self, x, y)
		local nodeClass = self:gettype().type
		assert(nodeClass:getmethod("eval"),
			string.format("Texture node type %s must have an eval method", tostring(nodeClass)))
		-- evalPoint all of the inputs, then pass the results to eval.
		local inputs = getInputEntries(nodeClass)
		local inputResults = inputs:map(function(e) return `self.[e.field]:evalPoint(x,y) end)
		return ensureVecEval(`self:eval(x, y, [inputResults]), nodeClass)
	end)

	-- Generate code for the virtual 'evalImage' method, given an 'eval' method
	local genEvalImage = macro(function(self, xres, yres, xlo, xhi, ylo, yhi)
		local nodeClass = self:gettype().type
		assert(nodeClass:getmethod("eval"),
			string.format("Texture node type %s must have an eval method", tostring(nodeClass)))
		-- Fetch an image to use for our output.
		-- evalImage all of the inputs, then iterate over the results, calling eval.
		-- Release the images used for input results.
		local inputs = getInputEntries(nodeClass)
		local inputResults = inputs:map(function(e) return `self.[e.field]:evalImage(xres,yres,xlo,ylo,yhi) end)
		local inputTemps = inputs:map(function(e) return symbol(e.type) end)
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
					outimg(x,y) = [ensureVecEval(`self:eval(xval, yval, [inputTempsXY(xval, yval)]), nodeClass)]
					xval = xval + xdelta
				end
				yval = yval + ydelta
			end
			[freeInputResults]
		in
			outimg
		end
	end)

	function NodeT.Metatype(nodeClass)

		terra nodeClass:evalPoint(x: real, y: real) : OutputType
			return genEvalPoint(self, x,y)
		end
		inherit.virtual(nodeClass, "evalPoint")

		terra nodeClass:evalImage(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
								  : &image.Image(real, nchannels)
			return genEvalImage(self, xres,yres,xlo,xhi,ylo,yhi)
		end
		inherit.virtual(nodeClass, "evalImage")

	end


	return NodeT

end)




return Node





