local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")


-- Alpha compositing with a separate mask as the alpha channel
-- (Can be applied to both grayscale and color inputs)
local MaskNode = node.makeNodeFromFunc("MaskNode", function(real, nchannels, GPU)
	local Color = Vec(real, nchannels, GPU)
	local Color1 = Vec(real, 1, GPU)
	return terra(bottom: Color, top: Color, mask: Color1)
		var alpha = mask(0)
		var result = (1.0 - alpha)*bottom + alpha*top
		escape
			if nchannels == 4 then
				emit quote result(3) = bottom(3) end	-- Preserve bottom alpha
			end
		end
		return result
	end, {nchannels, nchannels, 1}
end)


local Mask = S.memoize(function(real, nchannels, GPU)

	local BaseFunction = Function(real, nchannels, GPU)
	local Mask = BaseFunction.makeDefaultSubtype(
	"Mask",
	{
		{bottom = nchannels},
		{top = nchannels},
		{mask = 1}
	},
	{
	})

	terra Mask:expand(coordNode: &Mask.CoordNode) : &Mask.OutputNode
		return [MaskNode(real, GPU)].alloc():init(self.registers,
			self.bottom:expand(coordNode), self.top:expand(coordNode), self.mask:expand(coordNode))
	end
	inherit.virtual(Mask, "expand")

	return Mask

end)


return
{
	Mask = Mask
}