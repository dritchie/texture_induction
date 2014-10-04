local S = terralib.require("qs.lib.std")
local mathlib = terralib.require("utils.mathlib")
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuda.cuimage")
local Program = terralib.require("tex.program")
local Registers = terralib.require("tex.registers")
local randTables = terralib.require("tex.randTables")
local fns = terralib.require("tex.functions.functions")
local Vec = terralib.require("utils.linalg.vec")
local autoptr = terralib.require("utils.autopointer")


----------------------------------------------------------------------


local IMG_SIZE = 256
local GPU = true


----------------------------------------------------------------------

-- Generating some random textures from the simple expression tree grammar

local qs = terralib.require("qs")
local grammar = terralib.require("inference.grammar")
local colorTexGenModule = grammar(4, GPU)

-- Do this or no?
qs.initrand()

-- Set up registers
local registers = global(Registers(double, GPU))
local hostRegisters = global(Registers(double, false))
local terra initglobals()
	registers:init()
	hostRegisters:init()
end
initglobals()


-- The probabilistic program
local p = qs.program(function()

	-- Constants
	local FACTOR_WEIGHT = 1000.0

	-- Modules
	local colorTexGen = colorTexGenModule:open()
	local hostmath = mathlib(false)

	-- Define image types
	local HostImage = image.Image(qs.real, 4)
	local Image = HostImage
	if GPU then
		Image = CUDAImage(qs.real, 4)
	end

	-- The target image
	local targetImg = global(HostImage)
	local terra initglobals()
		targetImg:init(image.Format.PNG, "exampleTextures/256/016-r.png")
	end
	initglobals()

	-- Spatial domain MSE
	-- TODO: Do this computation through CUDA
	local terra mse(im1: &HostImage, im2: &HostImage)
		var sqerr = qs.real(0.0)
		for y=0,im1.height do
			for x=0,im2.height do
				-- Disregard alpha in difference computation
				var diffR = im1(x,y)(0) - im2(x,y)(0)
				var diffG = im1(x,y)(1) - im2(x,y)(1)
				var diffB = im1(x,y)(2) - im2(x,y)(2)
				sqerr = sqerr + diffR*diffR + diffG*diffG + diffB*diffB
			end
		end
		return hostmath.sqrt(sqerr / (im1.width*im1.height))
	end

	-- Likelihood term is just spatial domain MSE, for now.
	-- If we're rendering on the GPU, then (for now) we have to copy
	--    the image back to host memory to do the comparison.
	local likelihood = macro(function(tex)
		if not GPU then
			return quote qs.factor(-mse(tex, &targetImg) * FACTOR_WEIGHT) end
		else
			return quote
				var hosttex = hostRegisters.vec4Registers:fetch(IMG_SIZE, IMG_SIZE)
				tex:toHostImg(hosttex)
				qs.factor(-mse(hosttex, &targetImg) * FACTOR_WEIGHT)
				hostRegisters.vec4Registers:release(hosttex)
			end
		end
	end)

	return terra()
		var rootFn = colorTexGen(&registers)
		var program = [Program(qs.real, 4, GPU)].salloc():init(&registers, rootFn)
		var tex = registers.vec4Registers:fetch(IMG_SIZE, IMG_SIZE)
		program:interpretVector(tex, -0.5, 0.5, -0.5, 0.5)
		likelihood(tex)
		registers.vec4Registers:release(tex)
		return autoptr.create(rootFn)
	end

end)


-- Running inference / processing results
local doinference = qs.infer(p, qs.MAP,
	qs.MCMC(qs.TraceMHKernel(), {numsamps=2000, verbose=true})
)
local terra go()
	var rootFn = doinference()
	var program = [Program(qs.real, 4, GPU)].salloc():init(&registers, rootFn.ptr)
	program:ssaPrintPretty()
	var tex = registers.vec4Registers:fetch(IMG_SIZE, IMG_SIZE)
	program:interpretVector(tex, -0.5, 0.5, -0.5, 0.5)
	-- TODO: delete rootFn now that we're done with it?
	escape
		if not GPU then
			emit quote [image.Image(double, 4).save(uint8)](tex, image.Format.PNG, "randomTex.png") end
		else
			emit quote
				var img = [image.Image(double, 4)].salloc():init()
				tex:toHostImg(img)
				[image.Image(double, 4).save(uint8)](img, image.Format.PNG, "randomTex.png")
			end
		end
	end
	registers.vec4Registers:release(tex)
	-- Remove alpha channel from resulting image so we can see it better
	S.system("convert randomTex.png -alpha off randomTex.png")
end
go()

----------------------------------------------------------------------

-- -- Manually generating example textures

-- local OUT_NCHANNELS = 1

-- local Mat = terralib.require("utils.linalg.mat")
-- local Mat3 = Mat(double, 3, 3)
-- local RGBAColor = Vec(double, 4)

-- local gradients = randTables.const_gradients(double, GPU)
-- local registers = global(Registers(double, GPU))
-- local program = global(Program(double, OUT_NCHANNELS, GPU))
-- local terra initGlobals()
-- 	registers:init()

-- 	var perlin1 = [fns.Perlin(double, GPU)].alloc():init(&registers, gradients, 1.0, 3.0, 0.75, 0, 6)
-- 	var stretched = [fns.Transform(double, 1, GPU)].alloc():init(&registers, perlin1, Mat3.scale(10.0, 1.0))
-- 	var perlin2 = [fns.Perlin(double, GPU)].alloc():init(&registers, gradients, 1.0, 3.0, 0.75, 0, 2)
-- 	var warped = [fns.Warp(double, 1, GPU)].alloc():init(&registers, stretched, perlin2, 0.05)
-- 	var knots = [S.Vector(double)].salloc():init()
-- 	knots:insert(0.0)
-- 	knots:insert(1.0)
-- 	var colors = [S.Vector(RGBAColor)].salloc():init()
-- 	colors:insert(RGBAColor.create(115.0/255, 50.0/255, 18.0/255, 1.0))
-- 	colors:insert(RGBAColor.create(232.0/255, 115.0/255, 42.0/255, 1.0))
-- 	var colorized = [fns.Colorize(double, GPU)].alloc():init(&registers, warped, @knots, @colors)
-- 	var decolorized = [fns.Decolorize(double, GPU)].alloc():init(&registers, colorized)

-- 	program:init(&registers, decolorized)
-- 	-- program:treePrintPretty()
-- 	-- program:ssaPrintPretty()
-- end
-- initGlobals()


-- local t0 = terralib.currenttimeinseconds()
-- local compiledfn = Program(double, OUT_NCHANNELS, GPU).methods.compile(program:getpointer())
-- local t1 = terralib.currenttimeinseconds()
-- compiledfn:compile()
-- local t2 = terralib.currenttimeinseconds()
-- print("Specialization time: ", t1-t0)
-- print("Typechecking/compilation time: ", t2-t1)
-- print("Total time: ", t2-t0)


-- local outRegisters = macro(function()
-- 	if OUT_NCHANNELS == 1 then
-- 		return `registers.vec1Registers
-- 	else
-- 		return `registers.vec4Registers
-- 	end
-- end)

-- local terra test()
-- 	var tex = outRegisters():fetch(IMG_SIZE, IMG_SIZE)

-- 	-- program:interpretScalar(tex, 0.0, 1.0, 0.0, 1.0)
-- 	-- program:interpretVector(tex, 0.0, 1.0, 0.0, 1.0)
-- 	compiledfn(&program, tex, 0.0, 1.0, 0.0, 1.0)

-- 	escape
-- 		if not GPU then
-- 			emit quote [image.Image(double, OUT_NCHANNELS).save(uint8)](tex, image.Format.PNG, "perlinTest.png") end
-- 		else
-- 			emit quote
-- 				var img = [image.Image(double, OUT_NCHANNELS)].salloc():init()
-- 				tex:toHostImg(img)
-- 				[image.Image(double, OUT_NCHANNELS).save(uint8)](img, image.Format.PNG, "perlinTest_CUDA.png")
-- 			end
-- 		end
-- 	end

-- 	outRegisters():release(tex)

-- end
-- test()

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

-- -- How long does it take to compile a (reasonably-sized) kernel?

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


