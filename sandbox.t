local S = terralib.require("qs.lib.std")
local mathlib = terralib.require("utils.mathlib")
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local Program = terralib.require("tex.program")
local Registers = terralib.require("tex.registers")
local randTables = terralib.require("tex.randTables")
local fns = terralib.require("tex.functions.functions")


-- -- For reference:
-- local DEFAULT_PERLIN_FREQUENCY = `1.0
-- local DEFAULT_PERLIN_LACUNARITY = `2.0
-- local DEFAULT_PERLIN_PERSISTENCE = `0.5
-- local DEFAULT_PERLIN_OCTAVE_COUNT = `6

local IMG_SIZE = 256
local GPU = true


----------------------------------------------------------------------

-- -- Inferring parameters of Perlin noise

-- local qs = terralib.require("qs")

-- local p = qs.program(function()

-- 	local FACTOR_WEIGHT = 250.0

-- 	local gradients = randTables.const_gradients(qs.real, GPU)
-- 	local registers = global(Registers(qs.real, GPU))

-- 	local Image = image.Image(qs.real, 1)
-- 	local targetImg = global(Image)
-- 	local testImg = global(Image)

-- 	local terra initglobals()
-- 		registers:init()
-- 		targetImg:init(image.Format.PNG, "perlinTest_orig.png")
-- 		testImg:init()
-- 	end
-- 	initglobals()

-- 	-- For working with a compiled program
-- 	-- (For now, we just compile once. In general, we'll need a mechanism to trigger recompilation
-- 	--    whenever program structure changes)
-- 	local OutImgType = Image
-- 	if GPU then OutImgType = CUDAImage(qs.real, 1) end
-- 	local globalProgram = nil
-- 	local globalProgramFnPtr = global({&Program(qs.real, 1, GPU), &OutImgType, qs.real, qs.real, qs.real, qs.real}->{}, 0)
-- 	local function compileProgram(program)
-- 		globalProgram = Program(qs.real, 1, GPU).methods.compile(program)
-- 		globalProgramFnPtr:set(globalProgram:getpointer())
-- 	end

-- 	-- TODO: CUDA parallel reduction (the computation is almost negligible, actually, but all
-- 	--    the device->host memcpy's that we have to do when generating textures on the GPU
-- 	--    are cutting down the performance increase by about 2x)
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

-- 		var program = [Program(qs.real, 1, GPU)].salloc():init(&registers)
-- 		var perlin = [fns.PerlinNode(qs.real, GPU)].create(&registers, program:getInputCoordNode(),
-- 						gradients, frequency, lacunarity, persistence, 0, octaves)
-- 		program:setOuputNode(perlin)
-- 		var tex = registers.vec1Registers:fetch(IMG_SIZE, IMG_SIZE)

-- 		-- program:interpretScalar(tex, 0.0, 1.0, 0.0, 1.0)
-- 		-- program:interpretVector(tex, 0.0, 1.0, 0.0, 1.0)
-- 		if globalProgramFnPtr == nil then
-- 			compileProgram(program)
-- 		end
-- 		globalProgramFnPtr(program, tex, 0.0, 1.0, 0.0, 1.0)

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

-- 		registers.vec1Registers:release(tex)
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

-- Generating some noise

local Mat = terralib.require("utils.linalg.mat")
local Mat3 = Mat(double, 3, 3, GPU)

local gradients = randTables.const_gradients(double, GPU)
local registers = global(Registers(double, GPU))
local program = global(Program(double, 1, GPU))
local terra initGlobals()
	registers:init()

	var srcNoise = [fns.Perlin(double, GPU)].alloc():init(&registers, gradients, 1.0, 3.0, 0.75, 0, 6)
	var stretchedNoise = [fns.Transform(double, 1, GPU)].alloc():init(&registers, srcNoise, Mat3.scale(10.0, 1.0))
	var warpField = [fns.Perlin(double, GPU)].alloc():init(&registers, gradients, 1.0, 3.0, 0.75, 0, 2)
	var woodNoise = [fns.Warp(double, 1, GPU)].alloc():init(&registers, stretchedNoise, warpField, 0.05)

	program:init(&registers, woodNoise)
end
initGlobals()

local t0 = terralib.currenttimeinseconds()
local compiledfn = Program(double, 1, GPU).methods.compile(program:getpointer())
local t1 = terralib.currenttimeinseconds()
compiledfn:compile()
local t2 = terralib.currenttimeinseconds()
print("Specialization time: ", t1-t0)
print("Typechecking/compilation time: ", t2-t1)
print("Total time: ", t2-t0)


local terra test()
	var tex = registers.vec1Registers:fetch(IMG_SIZE, IMG_SIZE)

	-- program:interpretScalar(tex, 0.0, 1.0, 0.0, 1.0)
	-- program:interpretVector(tex, 0.0, 1.0, 0.0, 1.0)
	compiledfn(&program, tex, 0.0, 1.0, 0.0, 1.0)

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

	registers.vec1Registers:release(tex)

end
test()

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


----------------------------------------------------------------------

-- -- Testing some Terra -> Lua value conversions that are needed during texture
-- --    program compilation.

-- local struct Foo(S.Object) { x: int }
-- local struct Bar(S.Object) { x: int }

-- local terra makeFoo()
-- 	return Foo.alloc():init()
-- end

-- local f = makeFoo()
-- local b1 = terralib.cast(&Bar, f)
-- local b2 = terralib.cast(&Bar, f)
-- print(f)
-- print(b1)
-- print(b2)
-- print("f == b1", f == b1)
-- print("f == b2", f == b2)
-- local tbl = {}
-- tbl[f] = true
-- print("tbl[b1]", tbl[b1])

-- -- So casted cdata objects are considered equal, but they don't match
-- --    as table keys.

-- local struct Baz(S.Object) { foo: &Foo }

-- local terra makeBaz()
-- 	var baz = Baz.alloc():init()
-- 	baz.foo = Foo.alloc():init()
-- 	return baz
-- end

-- local bz = makeBaz()
-- local f1 = bz.foo
-- local f2 = bz.foo
-- print(f1)
-- print(f2)
-- print("f1 == f2", f1 == f2)
-- tbl = {}
-- tbl[f1] = true
-- print("tbl[f2]", tbl[f2])

-- -- So even two cdata objects corresponding to the same pointer don't
-- --    match as table keys. Bummer.

-- local HashMap = terralib.require("qs.lib.hashmap")
-- local hmap = terralib.new(HashMap(&opaque, bool))
-- hmap:__init()
-- hmap:put(f1, true)
-- local resultptr = hmap:getPointer(f2)
-- print(resultptr)

-- -- Ok, it works if I used a terra HashMap. But how do I extract the value of the result...
-- -- Wait...shit, no I need the map to go to *symbols*, which I can't store in a HashMap.

-- -- local u1 = tonumber(terralib.cast(uint64, f1))
-- -- local u2 = tonumber(terralib.cast(uint64, f2))
-- -- local u1 = tostring(terralib.cast(uint64, f1))
-- -- local u2 = tostring(terralib.cast(uint64, f2))
-- local u1 = tostring(f1)
-- local u2 = tostring(f2)
-- print(u1)
-- print(u2)
-- print("u1 == u2", u1 == u2)
-- tbl = {}
-- tbl[u1] = true
-- print("tbl[u2]", tbl[u2])

-- -- OK, I think this is the best I can do. tonumber isn't safe, because it incurs precision loss.
-- -- tostring is fine; just need to decide if i want to cast to uint64 first...

-- print(type(f1.x))


