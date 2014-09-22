local S = terralib.require("qs.lib.std")
local mathlib = terralib.require("utils.mathlib")
local image = terralib.require("utils.image")
local Program = terralib.require("tex.program")
local Registers = terralib.require("tex.registers")
local PerlinNode = terralib.require("tex.nodes.perlin")
local randTables = terralib.require("tex.randTables")


-- -- For reference:
-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6

local IMG_SIZE = 256
local GPU = true

----------------------------------------------------------------------

-- Inferring parameters of Perlin noise

local qs = terralib.require("qs")

local p = qs.program(function()

	local FACTOR_WEIGHT = 250.0

	local gradients = randTables.const_gradients(qs.real, GPU)
	local registers = global(Registers(qs.real, GPU))

	local Image = image.Image(qs.real, 1)
	local targetImg = global(Image)
	local testImg = global(Image)

	local terra initglobals()
		registers:init()
		targetImg:init(image.Format.PNG, "perlinTest.png")
		testImg:init()
	end
	initglobals()

	-- TODO: CUDA parallel reduction (the computation is almost negligible, actually, but all
	--    the device->host memcpy's that we have to do when generating textures on the GPU
	--    are cutting down the performance increase by about 2x)
	local mlib = mathlib(false)
	local terra imMSE(im1: &Image, im2: &Image)
		var sqerr = qs.real(0.0)
		for y=0,im1.height do
			for x=0,im1.width do
				var diff = im1(x,y) - im2(x,y)
				sqerr = sqerr + diff:dot(diff)
			end
		end
		return mlib.sqrt(sqerr / (im1.width*im1.height))
	end

	return terra()
		var frequency = qs.gammamv(1.0, 0.5, {struc=false})
		var lacunarity = qs.gammamv(2.0, 1.0, {struc=false})
		var persistence = qs.betamv(0.5, 0.05, {struc=false})
		var octaves = qs.poisson(6)

		var program = [Program(qs.real, 1, GPU)].salloc():init(&registers)
		var perlin = [PerlinNode(qs.real, GPU)].salloc():init(&registers, gradients,
			frequency, lacunarity, persistence, octaves)
		perlin:setInputCoordNode(program:getInputCoordNode())
		program:setOuputNode(perlin)
		var tex = registers.grayscaleRegisters:fetch(IMG_SIZE, IMG_SIZE)

		-- program:interpretScalar(tex, 0.0, 1.0, 0.0, 1.0)
		program:interpretVector(tex, 0.0, 1.0, 0.0, 1.0)

		escape
			if not GPU then
				emit quote qs.factor(-imMSE(tex, &targetImg) * FACTOR_WEIGHT) end
			else
				emit quote
					tex:toHostImg(&testImg)
					qs.factor(-imMSE(&testImg, &targetImg) * FACTOR_WEIGHT)
				end
			end
		end

		registers.grayscaleRegisters:release(tex)
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

-- -- Generating some noise

-- local gradients = randTables.const_gradients(double, GPU)
-- local terra test()
	
-- 	var registers = [Registers(double, GPU)].salloc():init()
-- 	var program = [Program(double, 1, GPU)].salloc():init(registers)
-- 	var perlin = [PerlinNode(double, GPU)].salloc():init(registers, gradients,
-- 													1.0, 3.0, 0.75, 6)
-- 	perlin:setInputCoordNode(program:getInputCoordNode())
-- 	program:setOuputNode(perlin)
-- 	var tex = registers.grayscaleRegisters:fetch(IMG_SIZE, IMG_SIZE)
-- 	-- program:interpretScalar(tex, 0.0, 1.0, 0.0, 1.0)
-- 	program:interpretVector(tex, 0.0, 1.0, 0.0, 1.0)
-- 	escape
-- 		if not GPU then
-- 			emit quote [image.Image(double, 1).save(uint8)](tex, image.Format.PNG, "perlinTest.png") end
-- 		else
-- 			emit quote
-- 				var img = [image.Image(double, 1)].salloc():init()
-- 				tex:toHostImg(img)
-- 				[image.Image(double, 1).save(uint8)](img, image.Format.PNG, "perlinTest_CUDA.png")
-- 			end
-- 		end
-- 	end
-- 	registers.grayscaleRegisters:release(tex)

-- end
-- test()

----------------------------------------------------------------------

-- -- How long does it take to compile a (reasonably-sized) kernel?

-- -- local Vec = terralib.require("utils.linalg.vec")
-- -- local custd = terralib.require("utils.cuda.custd")
-- -- local lerp = macro(function(lo, hi, t)
-- -- 	return `(1.0-t)*lo + t*hi
-- -- end)

-- local numNoiseSources = 1

-- local function makeKernel()
-- 	-- local real = double
-- 	-- local OutputType = Vec(real, 1, GPU)
-- 	-- local GradientTable = randTables.GradientTable(real, GPU)
-- 	-- local terra kernel(output: &OutputType, width: uint, height: uint, pitch: uint64, xlo: real, xhi: real, ylo: real, yhi: real,
-- 	-- 				   grads: GradientTable, freq: real, lac: real, pers: real, oct: uint)
-- 	-- 	var xi = custd.threadIdx.x()
-- 	-- 	var yi = custd.blockIdx.x()
-- 	-- 	var xt = xi / real(width)
-- 	-- 	var yt = yi / real(height)
-- 	-- 	var x = lerp(xlo, xhi, xt)
-- 	-- 	var y = lerp(ylo, yhi, yt)
-- 	-- 	var outptr = [&OutputType]( [&uint8](output) + yi*pitch ) + xi
-- 	-- 	(@outptr)(0) = 0.0
-- 	-- 	escape
-- 	-- 		for i=1,numNoiseSources do
-- 	-- 			emit quote
-- 	-- 				@outptr = @outptr + @[Vec(real, 1, GPU)].salloc():init([PerlinNode(real, GPU)].eval(x, y, grads, freq, lac, pers, oct))
-- 	-- 			end 
-- 	-- 		end
-- 	-- 	end
-- 	-- 	(@outptr) = (@outptr) / real(numNoiseSources)
-- 	-- end
-- 	local terra kernel() end
-- 	local K = terralib.cudacompile({kernel = kernel}, false)
-- 	return K.kernel
-- end

-- local globalkernel = nil
-- local function compileGlobalKernel()
-- 	globalkernel = makeKernel()
-- end

-- local numiters = 1
-- local gettime = terralib.cast({}->double, terralib.currenttimeinseconds)
-- local terra timeKernelCompilation()
-- 	-- Initialize it first
-- 	compileGlobalKernel()
-- 	var t0 = gettime()
-- 	for i=0,numiters do
-- 		compileGlobalKernel()
-- 	end
-- 	var t1 = gettime()
-- 	return (t1 - t0) / numiters
-- end
-- local time = timeKernelCompilation()
-- print(time)





