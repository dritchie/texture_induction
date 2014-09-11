local image = terralib.require("utils.image")
local PerlinNode = terralib.require("tex.nodes.perlin")
local ImagePool = terralib.require("tex.imagePool")
local randTables = terralib.require("tex.randTables")


-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6

local gradients = randTables.const_gradients(double)
local terra test()
	
	var impool = [ImagePool(double, 1)].salloc():init()
	var perlin = [PerlinNode(double)].salloc():init(impool, &gradients,
													1.0, 3.0, 0.75, 6)
	var tex = perlin:genTexturePointwise(512, 512, 0.0, 1.0, 0.0, 1.0)
	-- var tex = perlin:genTextureBlocked(512, 512, 0.0, 1.0, 0.0, 1.0)
	[image.Image(double, 1).save(uint8)](tex, image.Format.PNG, "perlinTest.png")
	impool:release(tex)

end
-- test:compile()
test()