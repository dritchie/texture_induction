local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local Registers = terralib.require("tex.registers")
local randTables = terralib.require("tex.randTables")
local Node = terralib.require("tex.nodes.node")
local noise = terralib.require("tex.noise")

local GradientTable = randTables.GradientTable


-- Perlin noise generator
-- Based on libnoise's perlin class (http://libnoise.sourceforge.net/index.html)


-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6


-- Perlin noise texture node
local PerlinNode = S.memoize(function(real, GPU)

	local Coord = Vec(real, 2, GPU)

	-- IMPORTANT: The perisistance parameter should be less than 1 to keep
	--    the output of this node in the range (0, 1)

	local struct PerlinNode(S.Object)
	{
		gradients: GradientTable(real, GPU),
		frequency: real,
		lacunarity: real,
		persistence: real,
		octaves: uint
	}
	local ParentNodeType = Node(real, 1, GPU)
	inherit.dynamicExtend(ParentNodeType, PerlinNode)
	ParentNodeType.Metatype(PerlinNode)

	terra PerlinNode:__init(registers: &Registers(real, GPU), grads: GradientTable(real, GPU),
							freq: real, lac: real, pers: real, oct: uint) : {}
		ParentNodeType.__init(self, registers) -- *must* be called
		ParentNodeType.initInputs(self)        -- *must* be called
		self.gradients = grads
		self.frequency = freq
		self.lacunarity = lac
		self.persistence = pers
		self.octaves = oct
	end

	PerlinNode.methods.eval = terra(coord: Coord, gradients: GradientTable(real, GPU),
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

	return PerlinNode

end)


return PerlinNode




