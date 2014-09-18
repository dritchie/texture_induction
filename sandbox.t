local S = terralib.require("qs.lib.std")
local mathlib = terralib.require("utils.mathlib")
local image = terralib.require("utils.image")
local PerlinNode = terralib.require("tex.nodes.perlin")
local ImagePool = terralib.require("tex.imagePool")
local randTables = terralib.require("tex.randTables")


-- -- For reference:
-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6

local IMG_SIZE = 256
local GPU = true

----------------------------------------------------------------------


-- local qs = terralib.require("qs")

-- local p = qs.program(function()

-- 	local FACTOR_WEIGHT = 250.0

-- 	local gradients = randTables.const_gradients(qs.real, GPU)
-- 	local impool = global(ImagePool(qs.real, 1, GPU))

-- 	local Image = image.Image(qs.real, 1)
-- 	local targetImg = global(Image)
-- 	local testImg = global(Image)

-- 	local terra initglobals()
-- 		impool:init()
-- 		targetImg:init(image.Format.PNG, "perlinTest.png")
-- 		testImg:init()
-- 	end
-- 	initglobals()

-- 	-- TODO: CUDA parallel reduction
-- 	local mlib = mathlib(false)
-- 	local terra imMSE(im1: &Image, im2: &Image)
-- 		var sqerr = qs.real(0.0)
-- 		for y=0,im1.height do
-- 			for x=0,im1.width do
-- 				var diff = im1(x,y) - im2(x,y)
-- 				sqerr = sqerr + diff:dot(diff)
-- 			end
-- 		end
-- 		return mlib.sqrt(sqerr / (im1.width*im1.height))
-- 	end

-- 	return terra()
-- 		var frequency = qs.gammamv(1.0, 0.5, {struc=false})
-- 		var lacunarity = qs.gammamv(2.0, 1.0, {struc=false})
-- 		var persistence = qs.betamv(0.5, 0.05, {struc=false})
-- 		var octaves = qs.poisson(6)

-- 		var perlin = [PerlinNode(qs.real, GPU)].salloc():init(&impool, gradients,
-- 			frequency, lacunarity, persistence, octaves)
-- 		-- var tex = perlin:interpretPixelwise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)
-- 		var tex = perlin:interpretNodewise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)

-- 		escape
-- 			if not GPU then
-- 				emit quote qs.factor(-imMSE(tex, &targetImg) * FACTOR_WEIGHT) end
-- 			else
-- 				emit quote
-- 					tex:toHostImg(&testImg)
-- 					qs.factor(-imMSE(&testImg, &targetImg) * FACTOR_WEIGHT)
-- 				end
-- 			end
-- 		end

-- 		impool:release(tex)
-- 		return frequency, lacunarity, persistence, octaves
-- 	end

-- end)

-- -- local doinference = qs.infer(p, qs.MAP, qs.MCMC(qs.TraceMHKernel(), {numsamps=2000, verbose=true}))
-- local doinference = qs.infer(p, qs.MAP, qs.MCMC(
-- 	qs.MixtureKernel({
-- 		-- qs.TraceMHKernel({doStruct=false}),
-- 		qs.DriftKernel(),
-- 		-- qs.HARMKernel(),
-- 		qs.TraceMHKernel({doNonstruct=false})
-- 	}, {0.75, 0.25}),
-- 	{numsamps=2000, verbose=true})
-- )

-- local terra report()
-- 	var frequency, lacunarity, persistence, octaves = doinference()
-- 	S.printf("frequency: %g, lacunarity: %g, persistence: %g, octaves: %u\n",
-- 		frequency, lacunarity, persistence, octaves)
-- end
-- report()


----------------------------------------------------------------------

local gradients = randTables.const_gradients(double, GPU)
local terra test()
	
	var impool = [ImagePool(double, 1, GPU)].salloc():init()
	var perlin = [PerlinNode(double, GPU)].salloc():init(impool, gradients,
													1.0, 3.0, 0.75, 6)
	-- var tex = perlin:interpretPixelwise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)
	var tex = perlin:interpretNodewise(IMG_SIZE, IMG_SIZE, 0.0, 1.0, 0.0, 1.0)
	escape
		if not GPU then
			emit quote [image.Image(double, 1).save(uint8)](tex, image.Format.PNG, "perlinTest.png") end
		else
			emit quote
				var img = [image.Image(double, 1)].salloc():init()
				tex:toHostImg(img)
				[image.Image(double, 1).save(uint8)](img, image.Format.PNG, "perlinTest_CUDA.png")
			end
		end
	end
	impool:release(tex)

end
test()



