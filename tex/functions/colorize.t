local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")



-- TODO: Store gradient points in constant memory instead?
local MAX_NUM_GRAD_POINTS = 10



local ColorizeNode = node.makeNodeFromFunc("ColorizeNode", function(real, GPU)
	local Vec1 = Vec(real, 1, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	local GradKnots = real[MAX_NUM_GRAD_POINTS]
	local GradColors = RGBAColor[MAX_NUM_GRAD_POINTS]
	local lerp = macro(function(lo, hi, t) return `(1.0-t)*lo + t*hi end)
	return terra(input: Vec1, knots: GradKnots, colors: GradColors, n: uint)
		var val = input(0)
		var outcolor : RGBAColor
		if n == 1 then
			outcolor = colors[0]
		else
			for i=0,n do
				-- TODO: How much branch divergence does this cause?
				if knots[i] < val then
					outcolor = lerp(colors[i], colors[i+1], (val-knots[i])/(knots[i+1]-knots[i]))
					break
				end
			end
		end
		return outcolor
	end, {1}
end)


local Colorize = S.memoize(function(real, GPU)

	local RGBAColor = Vec(real, 4, GPU)
	local GradKnots = real[MAX_NUM_GRAD_POINTS]
	local GradColors = RGBAColor[MAX_NUM_GRAD_POINTS]

	local BaseFunction = Function(real, 4, GPU)
	local Colorize = BaseFunction.makeDefaultSubtype(
	"Colorize",
	{
		{input = 1}
	},
	{
		{knots = S.Vector(real)},
		{colors = S.Vector(RGBAColor)},
	})

	terra Colorize:expand(coordNode: &Colorize.CoordNode) : &Colorize.OutputNode
		if self.knots:size() ~= self.colors:size() then
			S.printf("Colorize: Must have same number of knots and colors\n")
			S.assert(false)
		end
		var n = self.knots:size()
		if n == 0 then
			S.printf("Colorize: no gradient points provided.\n")
			S.assert(false)
		end
		if n > MAX_NUM_GRAD_POINTS then
			S.printf("Colorize: Too many gradient points (maximum number is %d)\n", MAX_NUM_GRAD_POINTS)
			S.assert(false)
		end
		var _knots : GradKnots
		var _colors : GradColors
		for i=0,n do
			_knots[i] = self.knots(i)
			_colors[i] = self.colors(i)
		end
		return [ColorizeNode(real, GPU)].alloc():init(self.registers, self.input:expand(coordNode),
													  _knots, _colors, n)
	end
	inherit.virtual(Colorize, "expand")

	return Colorize

end)



return
{
	Colorize = Colorize
}