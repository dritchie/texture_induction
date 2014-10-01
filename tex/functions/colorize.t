local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")
local Function = terralib.require("tex.functions.function")
local inherit = terralib.require("utils.inheritance")
local util = terralib.require("utils.util")



-- TODO: Store gradient points in constant memory instead?
local MAX_NUM_GRAD_POINTS = 8



local ColorizeNode = node.makeNodeFromFunc("ColorizeNode", function(real, GPU)
	local Vec1 = Vec(real, 1, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	local GradKnots = real[MAX_NUM_GRAD_POINTS]
	local GradColors = RGBAColor[MAX_NUM_GRAD_POINTS]
	local lerp = macro(function(lo, hi, t) return `(1.0-t)*lo + t*hi end)
	return terra(input: Vec1, knots: GradKnots, colors: GradColors, n: uint)
		var val = input(0)
		for i=0,n do
			-- TODO: How much branch divergence does this cause?
			if knots[i] < val then
				return lerp(colors[i], colors[i+1], (val-knots[i])/(knots[i+1]-knots[i]))
			end
		end
	end, {1}
end)


local Colorize = S.memoize(function(real, GPU)

	local RGBAColor = Vec(real, 4, GPU)
	local RGBAColor_cpu = Vec(real, 4)

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
		{colors = S.Vector(RGBAColor_cpu)},
	})
	Colorize.MAX_NUM_GRAD_POINTS = MAX_NUM_GRAD_POINTS

	terra Colorize:expand(coordNode: &Colorize.CoordNode) : &Colorize.OutputNode
		if self.knots:size() ~= self.colors:size() then
			S.printf("Colorize: Must have same number of knots and colors\n")
			S.assert(false)
		end
		var n = self.knots:size()
		if n < 2 then
			S.printf("Colorize: too few gradient points provided (need at least two, for t=0 and t=1).\n")
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
			-- coerce CPU color into platform-specific color
			_colors[i] = @[&RGBAColor](self.colors:get(i))
		end
		return [ColorizeNode(real, GPU)].alloc():init(self.registers, self.input:expand(coordNode),
													  _knots, _colors, n)
	end
	inherit.virtual(Colorize, "expand")

	terra Colorize:printAggParams(tablevel: uint) : {}
		util.printTabs(tablevel)
		S.printf("gradient:\n")
		for i=0,self.knots:size() do
			util.printTabs(tablevel+1)
			var col = self.colors(i)
			S.printf("t = %.2g -> %.2g %.2g %.2g %.2g\n", self.knots(i), col(0), col(1), col(2), col(3))
		end
	end
	inherit.virtual(Colorize, "printAggParams")

	return Colorize

end)



return
{
	Colorize = Colorize
}




