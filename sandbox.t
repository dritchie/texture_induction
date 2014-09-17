local S = terralib.require("qs.lib.std")
local image = terralib.require("utils.image")
local PerlinNode = terralib.require("tex.nodes.perlin")
local ImagePool = terralib.require("tex.imagePool")
local randTables = terralib.require("tex.randTables")
local tmath = terralib.require("qs.lib.tmath")


-- -- For reference:
-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6

local IMG_SIZE = 256
local GPU = false

----------------------------------------------------------------------


local qs = terralib.require("qs")

local p = qs.program(function()

	local FACTOR_WEIGHT = 250.0

	local gradients = randTables.const_gradients(qs.real)
	local impool = global(ImagePool(qs.real, 1, GPU))

	local Image = image.Image(qs.real, 1)
	local targetImg = global(Image)

	local terra initglobals()
		impool:init()
		targetImg:init(image.Format.PNG, "perlinTest.png")
	end
	initglobals()

	local terra imMSE(im1: &Image, im2: &Image)
		var sqerr = qs.real(0.0)
		for y=0,im1.height do
			for x=0,im1.width do
				var diff = im1(x,y) - im2(x,y)
				sqerr = sqerr + diff:dot(diff)
			end
		end
		return tmath.sqrt(sqerr / (im1.width*im1.height))
	end

	return terra()
		var frequency = qs.gammamv(1.0, 0.5, {struc=false})
		var lacunarity = qs.gammamv(2.0, 1.0, {struc=false})
		var persistence = qs.betamv(0.5, 0.05, {struc=false})
		var octaves = qs.poisson(6)

		var perlin = [PerlinNode(qs.real, GPU)].salloc():init(&impool, &gradients,
			frequency, lacunarity, persistence, octaves)
		var tex = perlin:interpretPixelwise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)
		-- var tex = perlin:interpretNodewise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)

		qs.factor(-imMSE(tex, &targetImg) * FACTOR_WEIGHT)

		impool:release(tex)
		return frequency, lacunarity, persistence, octaves
	end

end)

-- local doinference = qs.infer(p, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=2000, verbose=true}))
local doinference = qs.infer(p, qs.MAP, qs.MCMC(
	qs.MixtureKernel({
		-- qs.TraceMHKernel({doStruct=false}),
		qs.DriftKernel(),
		-- qs.HARMKernel(),
		qs.TraceMHKernel({doNonstruct=false})
	}, {0.75, 0.25}),
	{numsamps=2000, verbose=true})
)

local terra report()
	var frequency, lacunarity, persistence, octaves = doinference()
	S.printf("frequency: %g, lacunarity: %g, persistence: %g, octaves: %u\n",
		frequency, lacunarity, persistence, octaves)
end
report()


----------------------------------------------------------------------

-- local gradients = randTables.const_gradients(double)
-- local terra test()
	
-- 	var impool = [ImagePool(double, 1, GPU)].salloc():init()
-- 	var perlin = [PerlinNode(double, GPU)].salloc():init(impool, &gradients,
-- 													1.0, 3.0, 0.75, 6)
-- 	var tex = perlin:interpretPixelwise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)
-- 	-- var tex = perlin:interpretNodewise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)
-- 	[image.Image(double, 1).save(uint8)](tex, image.Format.PNG, "perlinTest.png")
-- 	impool:release(tex)

-- end
-- test()



