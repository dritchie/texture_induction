local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")


-- Alpha compositing
-- TODO: Additional blend modes (e.g. Multiply, Screen)?
local BlendNode = node.makeNodeFromFunc("BlendNode", function(real, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	return terra(bottom: RGBAColor, top: RGBAColor, opacity: real)
		var alpha = top(3)*opacity
		var result = (1.0 - alpha)*bottom + alpha*top
		result(3) = bottom(3)	-- Preserve bottom alpha
		return result
	end, {4, 4}
end)


local Blend = S.memoize(function(real, GPU)

	local BaseFunction = Function(real, 4, GPU)
	local Blend = BaseFunction.makeDefaultSubtype(
	"Blend",
	{
		{bottom = 4},
		{top = 4}
	},
	{
		{opacity = real}
	})

	terra Blend:expand(coordNode: &Blend.CoordNode) : &Blend.OutputNode
		return [BlendNode(real, GPU)].alloc():init(self.registers,
			self.bottom:expand(coordNode), self.top:expand(coordNode), self.opacity)
	end
	inherit.virtual(Blend, "expand")

	return Blend

end)


return
{
	Blend = Blend
}