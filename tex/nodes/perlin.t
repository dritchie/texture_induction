local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local image = terralib.require("utils.image")
local inherit = terralib.require("utils.inheritance")
local ImagePool = terralib.require("tex.imagePool")
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


-- Perlin noise texture node
local PerlinNode = S.memoize(function(real)

	-- IMPORTANT: The perisistance parameter should be less than 1 to keep
	--    the output of this node in the range (0, 1)

	local struct PerlinNode(S.Object)
	{
		gradients: &GradientTable(real),
		frequency: real,
		lacunarity: real,
		persistence: real,
		octaves: uint
	}
	local Node = node.Node(real, 1)
	PerlinNode.ParentType = Node
	node.Metatype(PerlinNode)
	inherit.dynamicExtend(Node, PerlinNode)

	terra PerlinNode:__init(imPool: &ImagePool(real, 1), grads: &GradientTable(real),
							freq: real, lac: real, pers: real, oct: uint) : {}
		Node.__init(self, imPool)
		self.gradients = grads
		self.frequency = freq
		self.lacunarity = lac
		self.persistence = pers
		self.octaves = oct
	end

	terra PerlinNode:eval(x: real, y: real)
		x = x * self.frequency
		y = y * self.frequency
		var persist = real(1.0)
		var value = real(0.0)

		for octave=0,self.octaves do
			value = value + (persist * [noise.gradientCoherent(real)](x, y, octave, self.gradients))
			persist = persist * self.persistence
			x = x * self.lacunarity
			y = y * self.lacunarity
		end

		-- Transform value from (-1, 1) to (0, 1)
		-- S.printf("%g\n", (value + 1) * 0.5)
		return (value + 1) * 0.5
	end

	return PerlinNode

end)


return PerlinNode




