local S = require("qs.lib.std")
local Vec = require("utils.linalg.vec")
local randTables = require("tex.randTables")
local node = require("tex.functions.node")
local noise = require("tex.noise")
local Function = require("tex.functions.function")
local inherit = require("utils.inheritance")

local GradientTable = randTables.GradientTable


-- Perlin noise generator
-- Based on libnoise's perlin class (http://libnoise.sourceforge.net/index.html)


-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6


local PerlinNode = node.makeNodeFromFunc("PerlinNode", function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	return terra(coord: Coord, gradients: GradientTable(real, GPU),
				 frequency: real, lacunarity: real, persistence: real, startOctave: uint, nOctaves: uint)
		var x = coord(0)
		var y = coord(1)
		x = x * frequency
		y = y * frequency
		var persist = real(1.0)
		var value = real(0.0)

		for octave=startOctave,nOctaves do
			value = value + (persist * [noise.gradientCoherent(real, GPU)](x, y, octave, gradients))
			persist = persist * persistence
			x = x * lacunarity
			y = y * lacunarity
		end

		-- Transform value from (-1, 1) to (0, 1)
		return (value + 1) * 0.5
	end, {2}
end)


local Perlin = S.memoize(function(real, GPU)

	local BaseFunction = Function(real, 1, GPU)
	local Perlin = BaseFunction.makeDefaultSubtype(
	"Perlin",
	{},
	{
		{gradients = GradientTable(real, GPU)},
		{frequency = real},
		{lacunarity = real},
		{persistence = real},
		{startOctave = uint},
		{nOctaves = uint}
	})

	terra Perlin:expand(coordNode: &Perlin.CoordNode) : &Perlin.OutputNode
		return [PerlinNode(real, GPU)].alloc():init(self.registers, coordNode, self.gradients, self.frequency, 
								self.lacunarity, self.persistence, self.startOctave, self.nOctaves)
	end
	inherit.virtual(Perlin, "expand")

	return Perlin

end)


return
{
	Perlin = Perlin
}

