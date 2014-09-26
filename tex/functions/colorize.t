local S = terralib.require("qs.lib.std")
local node = terralib.require("tex.functions.node")
local Vec = terralib.require("utils.linalg.vec")


-- TODO: Store gradient points in constant memory instead?
local MAX_NUM_GRAD_POINTS = 10


local ColorizeNode = node.makeNodeFromFunc("ColorizeNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	local Vec1 = Vec(real, 1, GPU)
	local RGBColor = Vec(real, 3, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	local GradKnots = real[MAX_NUM_GRAD_POINTS]
	local GradColors = RGBAColor[MAX_NUM_GRAD_POINTS]
	local lerp = macro(function(lo, hi, t) return `(1.0-t)*lo + t*hi end)
	return terra(coord: Coord, input: Vec1, knots: GradColors, colors: GradColors, n: uint, alpha: real)
		var val = input(0)
		var outcolor_rgb : RGBColor
		for i=0,n do
			-- TODO: How much branch divergence does this cause?
			if knots[i] < val then
				outcolor_rgb = lerp(knots[i], knots[i+1], (val-knots[i])/(knots[i+1]-knots[i]))
				break
			end
		end
		return RGBAColor.create(outcolor_rgb(0), outcolor_rgb(1), outcolor_rgb(2), val*alpha)
	end, {1}
end)


-- Wrapper that allows use of standard Vectors to store grad info
local ColorizeNodeWrapper = S.memoize(function(real, GPU)
	local ColorizeNodeT = ColorizeNode(real, GPU)
	local rawCreate = ColorizeNodeT.methods.create
	local succ, T = rawCreate:peektype()
	local Registers = T.parameters[1]
	local CoordNode = T.parameters[2]
	local GrayscaleNode = T.parameters[3]
	local RGBColor = Vec(real, 3, GPU)
	local RGBAColor = Vec(real, 4, GPU)
	local GradKnots = real[MAX_NUM_GRAD_POINTS]
	local GradColors = RGBAColor[MAX_NUM_GRAD_POINTS]
	ColorizeNodeT.methods.create = terra(registers: Registers, coordNode: CoordNode, inputNode: GrayscaleNode,
										 knots: &S.Vector(real), colors: &S.Vector(RGBColor), alpha: real)
		if knots:size() ~= colors:size() then
			S.printf("ColorizeNode: Must have same number of knots and colors\n")
			S.assert(false)
		end
		var n = knots:size()
		if n > MAX_NUM_GRAD_POINTS then
			S.printf("ColorizeNode: Too many gradient points (maximum number is %d)\n", MAX_NUM_GRAD_POINTS)
			S.assert(false)
		end
		var _knots : GradKnots
		var _colors : GradColors
		for i=0,n do
			_knots[i] = knots(i)
			_colors[i] = colors(i)
		end
		return rawCreate(registers, coordNode, inputNode, _knots, _colors, n, alpha)
	end
	return ColorizeNodeT
end)


return
{
	ColorizeNode = ColorizeNodeWrapper
}