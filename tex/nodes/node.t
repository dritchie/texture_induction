local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local ImagePool = terralib.require("tex.imagePool")


-- Abstract base class for all texture nodes
local Node = S.memoize(function(real, nchannels)

	local OutputType = Vec(real, nchannels)

	local struct Node(S.Object)
	{
		imagePool: &ImagePool(real, nchannels)
	}
	Node.RealType = real
	Node.NumChannels = nchannels
	Node.OutputType = OutputType

	terra Node:__init(impool: &ImagePool(real, nchannels))
		self.imagePool = impool
	end

	-- Destructor does nothing, but it's virtual so that if subclass
	--    destructors need to do something, they can.
	terra Node:__destruct() : {} end
	inherit.virtual(Node, "__destruct")

	-- Evaluate the texture function represented by this Node at the point (x,y)
	inherit.purevirtual(Node, "evalPoint", {real,real}->OutputType)

	-- Evaluate the texture function over an entire image
	inherit.purevirtual(Node, "evalImage", {uint,uint,real,real,real,real}->&image.Image(real,nchannels))

	return Node

end)


-- Many texture nodes have input nodes; this utility allows easy creation of members for those
--    inputs, as well as getters/setters
local function defineInputs(nodeClass, numchannelsList)
	for i,nchannels in ipairs(numchannelsList) do
		local typ = &Node(nodeClass.RealType, nchannels)
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



-- Utilities for the standard metatype, below.

function stringStartsWith(str, prefix)
   return string.sub(str, 1, string.len(prefix)) == prefix
end

local function getInputEntries(nodeClass)
	local lst = terralib.newlist()
	for _,entry in ipairs(nodeClass.entries) do
		if stringStartsWith(entry.field, "input") then
			lst:insert(entry)
		end
	end
	return lst
end

local genEvalPoint = macro(function(self, x, y)
	local nodeClass = self:gettype().type
	assert(nodeClass:getmethod("evalPointImpl"),
		string.format("Texture node type %s must have an evalPointImpl method", tostring(nodeClass)))
	-- evalPoint all of the inputs, then pass the results to evalPointImpl.
	local inputs = getInputEntries(nodeClass)
	local inputResults = inputs:map(function(e) return `self.[e.field]:evalPoint(x,y) end)
	return `self:evalPointImpl(x, y, [inputResults])
end)

local genEvalImage = macro(function(self, xres, yres, xlo, xhi, ylo, yhi)
	local nodeClass = self:gettype().type
	assert(nodeClass:getmethod("evalPointImpl"),
		string.format("Texture node type %s must have an evalPointImpl method", tostring(nodeClass)))
	-- Fetch an image to use for our output.
	-- evalImage all of the inputs, then iterate over the results, calling evalPointImpl.
	-- Release the images used for input results.
	local inputs = getInputEntries(nodeClass)
	local inputResults = inputs:map(function(e) return `self.[e.field]:evalImage(xres,yres,xlo,ylo,yhi) end)
	local inputTemps = inputs:map(function(e) return symbol(e.type) end)
	local inputTempsAssign = #inputTemps > 0 and
		quote var [inputTemps] = [inputResults] end
	or
		quote end
	function inputTempsXY(x,y)
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
		var xval = xmin
		var yval = ymin
		for y=0,yres do
			for x=0,xres do
				outimg(x,y) = self:evalPointImpl(xval, yval, [inputTempsXY(xval, yval)])
				xval = xval + xdelta
			end
			yval = yval + ydelta
		end
		[freeInputResults]
	in
		outimg
	end
end)



-- A standard metatype for Nodes that defines the main interface functions in terms of a single
--    pointwise implementation that we assume to exist.
local function Metatype(nodeClass)

	-- First, always apply S.Object
	S.Object(nodeClass)

	local real = nodeClass.RealType

	terra nodeClass:evalPoint(x: real, y: real) : Node.OutputType
		return [genEvalPoint(x,y)]
	end
	inherit.virtual(nodeClass, "evalPoint")

	terra nodeClass:evalImage(xres: uint, yres: uint, xlo: real, xhi: real, ylo: real, yhi: real)
							  : &image.Image(real, nodeClass.NumChannels)
		return [genEvalImage(xres,yres,xlo,xhi,ylo,yhi)]
	end
	inherit.virtual(nodeClass, "evalImage")

end


return 
{
	Node = Node,
	defineInputs = defineInputs,
	Metatype = Metatype
}





