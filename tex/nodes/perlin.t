local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local randTables = terralib.require("tex.randTables")
local node = terralib.require("tex.nodes.node")
local noise = terralib.require("tex.noise")

local GradientTable = randTables.GradientTable


-- Perlin noise generator
-- Based on libnoise's perlin class (http://libnoise.sourceforge.net/index.html)


-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6


return node.makeNodeFromFunc(function(real, GPU)
	local Coord = Vec(real, 2, GPU)
	return terra(coord: Coord, gradients: GradientTable(real, GPU),
				 frequency: real, lacunarity: real, persistence: real, octaves: uint)
		var x = coord(0)
		var y = coord(1)
		x = x * frequency
		y = y * frequency
		var persist = real(1.0)
		var value = real(0.0)

		for octave=0,octaves do
			value = value + (persist * [noise.gradientCoherent(real, GPU)](x, y, octave, gradients))
			persist = persist * persistence
			x = x * lacunarity
			y = y * lacunarity
		end

		-- Transform value from (-1, 1) to (0, 1)
		return (value + 1) * 0.5
	end
end, {})




