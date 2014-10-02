local S = terralib.require("qs.lib.std")
local qs = terralib.require("qs")
local fns = terralib.require("tex.functions.functions")
local Function = terralib.require("tex.functions.function")
local Registers = terralib.require("tex.registers")
local randTables = terralib.require("tex.randTables")
local Vec = terralib.require("utils.linalg.vec")
local Mat = terralib.require("utils.linalg.mat")
local mathlib = terralib.require("utils.mathlib")
local inherit = terralib.require("utils.inheritance")

-- CPU math
local mlib = mathlib(false)

-- Incredibly simple grammar for generating texture programs.
-- Starts at the output function.
-- Generates each input independently.
-- Uses the same rules no matter what type of function it's currently expanding.
return function(nOutChannels, GPU)
	assert(nOutChannels == 1 or nOutChannels == 4,
		"grammar: nOutChannels must be either 1 (grayscale) or 4 (color)")
	return qs.module(function()

		-- Types / typedefs
		local Regs = Registers(qs.real, GPU)

		local Generator = S.memoize(function(nchannels)
			local struct Generator(S.Object) {}
			inherit.purevirtual(Generator, "generateImpl", {&Regs} ->{&Function(qs.real, nchannels, GPU)})
			Generator.methods.generate = qs.method(terra(self: &Generator, registers: &Regs)
				return self:generateImpl(registers)
			end)
			return Generator
		end)

		local GrayscaleGenerator = Generator(1)
		local ColorGenerator = Generator(4)


		-- Globals
		local grayscaleProbs = global(S.Vector(qs.real))
		local grayscaleGens = global(S.Vector(&GrayscaleGenerator))
		local colorProbs = global(S.Vector(qs.real))
		local colorGens = global(S.Vector(&ColorGenerator))


		-- The master generator function
		-- local fnsGenerated = global(uint, 0)
		local genFn = S.memoize(function(nchannels)
			local probs = nchannels == 4 and colorProbs or grayscaleProbs
			local gens = nchannels == 4 and colorGens or grayscaleGens
			return qs.func(terra(registers: &Regs)
				-- fnsGenerated = fnsGenerated + 1
				-- S.printf("functions generated: %u\n", fnsGenerated)
				var which = qs.categorical(&probs)
				return gens(which):generate(registers)
			end)
		end)

		
		-----------------------------------------------------
		-- Random generators for different function types  --
		-----------------------------------------------------

		local struct PerlinGenerator(S.Object) {}
		inherit.dynamicExtend(Generator(1), PerlinGenerator)
		local gradients = randTables.const_gradients(qs.real, GPU)
		terra PerlinGenerator:generateImpl(registers: &Regs) : &Function(qs.real, 1, GPU)
			var frequency = qs.gammamv(1.0, 2.0, {struc=false})
			var lacunarity = 1.0 + qs.gammamv(2.0, 3.0, {struc=false})
			var persistence = qs.uniform(0.0, 1.0, {struc=false})
			-- TODO: Should these be considered structural, or no?
			var startOctave = qs.poisson(1, {struc=false})
			var octaves = 1 + qs.poisson(6, {struc=false})
			return [fns.Perlin(qs.real, GPU)].alloc():init(registers, gradients,
				frequency, lacunarity, persistence, startOctave, octaves)
		end
		inherit.virtual(PerlinGenerator, "generateImpl")

		local TransformGenerator = S.memoize(function(nchannels)
			local Mat3 = Mat(qs.real, 3, 3)
			local struct TransformGenerator(S.Object) {}
			inherit.dynamicExtend(Generator(nchannels), TransformGenerator)
			terra TransformGenerator:generateImpl(registers: &Regs) : &Function(qs.real, nchannels, GPU)
				var input = [genFn(nchannels)](registers)
				-- TODO: Should we allow negative scales (i.e. reflections) as well?
				var scalex = mlib.exp(qs.gaussian(0.0, 1.0, {struc=false}))
				var scaley = mlib.exp(qs.gaussian(0.0, 1.0, {struc=false}))
				var ang = qs.gaussian(0.0, [math.pi/4.0], {struc=false})
				var xform = Mat3.rotate(ang) * Mat3.scale(scalex, scaley)
				return [fns.Transform(qs.real, nchannels, GPU)].alloc():init(registers, input, xform)
			end
			inherit.virtual(TransformGenerator, "generateImpl")
			return TransformGenerator
		end)

		local struct DecolorizeGenerator(S.Object) {}
		inherit.dynamicExtend(Generator(1), DecolorizeGenerator)
		terra DecolorizeGenerator:generateImpl(registers: &Regs) : &Function(qs.real, 1, GPU)
			var input = [genFn(4)](registers)
			return [fns.Decolorize(qs.real, GPU)].alloc():init(registers, input)
		end
		inherit.virtual(DecolorizeGenerator, "generateImpl")

		local WarpGenerator = S.memoize(function(nchannels)
			local struct WarpGenerator(S.Object) {}
			inherit.dynamicExtend(Generator(nchannels), WarpGenerator)
			terra WarpGenerator:generateImpl(registers: &Regs) : &Function(qs.real, nchannels, GPU)
				var input = [genFn(nchannels)](registers)
				var warpfield = [genFn(1)](registers)
				var strength = 0.08 * qs.gammamv(1.0, 4.0, {struc=false})
				return [fns.Warp(qs.real, nchannels, GPU)].alloc():init(registers, input, warpfield, strength)
			end
			inherit.virtual(WarpGenerator, "generateImpl")
			return WarpGenerator
		end)

		local MaskGenerator = S.memoize(function(nchannels)
			local struct MaskGenerator(S.Object) {}
			inherit.dynamicExtend(Generator(nchannels), MaskGenerator)
			terra MaskGenerator:generateImpl(registers: &Regs) : &Function(qs.real, nchannels, GPU)
				var bot = [genFn(nchannels)](registers)
				var top = [genFn(nchannels)](registers)
				var mask = [genFn(1)](registers)
				return [fns.Mask(qs.real, nchannels, GPU)].alloc():init(registers, bot, top, mask)
			end
			inherit.virtual(MaskGenerator, "generateImpl")
			return MaskGenerator
		end)

		local struct ColorizeGenerator(S.Object) {}
		inherit.dynamicExtend(Generator(4), ColorizeGenerator)
		local RGBAColor = Vec(qs.real, 4)
		-- Uniform prior over num points
		-- TODO: Better prior on num points
		-- TODO: Instead of num points, have a separate flip controlling the use/unuse of each point?
		local MAX_NUM_GRAD_POINTS = fns.Colorize(qs.real, GPU).MAX_NUM_GRAD_POINTS - 2
		local nPointsProbs = global(qs.real[MAX_NUM_GRAD_POINTS])
		for i=1,MAX_NUM_GRAD_POINTS do
			nPointsProbs:get()[i-1] = 1.0/MAX_NUM_GRAD_POINTS
		end
		terra ColorizeGenerator:generateImpl(registers: &Regs) : &Function(qs.real, 4, GPU)
			var input = [genFn(1)](registers)
			var knots = [S.Vector(qs.real)].salloc():init()
			var colors = [S.Vector(RGBAColor)].salloc():init()
			-- var npoints = qs.categorical(nPointsProbs) + 1
			var npoints = MAX_NUM_GRAD_POINTS
			var currKnot = qs.real(0.0)
			for i=0,npoints+2 do
				if i == 0 then
					knots:insert(0.0)
				elseif i == npoints+1 then
					knots:insert(1.0)
				else
					-- Generate knots by stick-breaking
					-- (Take care to ensure that knots will remain correct under any perturbation)
					var knotSpaceLeft = 1.0 - currKnot
					var knot = currKnot + (qs.uniform(0.0, 0.9, {struc=false}) * knotSpaceLeft)
					currKnot = knot
					knots:insert(knot)
				end
				-- Generate colors uniformly at random
				-- TODO: Make color prior encourage more smooth change / continuity?
				colors:insert(RGBAColor.create(qs.uniform(0.0, 1.0, {struc=false}),
											   qs.uniform(0.0, 1.0, {struc=false}),
											   qs.uniform(0.0, 1.0, {struc=false}),
											   qs.uniform(0.0, 1.0, {struc=false})))
											   -- 1.0))
			end
			return [fns.Colorize(qs.real, GPU)].alloc():init(registers, input, @knots, @colors)
		end
		inherit.virtual(ColorizeGenerator, "generateImpl")

		local struct BlendGenerator(S.Object) {}
		inherit.dynamicExtend(Generator(4), BlendGenerator)
		terra BlendGenerator:generateImpl(registers: &Regs) : &Function(qs.real, 4, GPU)
			var bot = [genFn(4)](registers)
			var top = [genFn(4)](registers)
			var opacity = qs.uniform(0.0, 1.0, {struc=false})
			return [fns.Blend(qs.real, GPU)].alloc():init(registers, bot, top, opacity)
		end
		inherit.virtual(BlendGenerator, "generateImpl")

		-- Initialize the global prob/gen lists
		local isinitialized = global(bool, 0)
		local terra init()
			isinitialized = true
			grayscaleProbs:init()
			grayscaleGens:init()
			colorProbs:init()
			colorGens:init()

			-- Fill in the global probs/gen lists
			
			grayscaleGens:insert(PerlinGenerator.alloc():init())
			-- grayscaleGens:insert(DecolorizeGenerator.alloc():init())
			grayscaleGens:insert([TransformGenerator(1)].alloc():init())
			grayscaleGens:insert([WarpGenerator(1)].alloc():init())
			grayscaleGens:insert([MaskGenerator(1)].alloc():init())
			
			colorGens:insert(ColorizeGenerator.alloc():init())
			colorGens:insert(BlendGenerator.alloc():init())
			colorGens:insert([TransformGenerator(4)].alloc():init())
			colorGens:insert([WarpGenerator(4)].alloc():init())
			colorGens:insert([MaskGenerator(4)].alloc():init())

			-- (Just use uniform probabilities for now)
			for i=0,grayscaleGens:size() do grayscaleProbs:insert(1.0) end
			grayscaleProbs(0) = 5.0
			for i=0,colorGens:size() do colorProbs:insert(1.0) end
			colorProbs(0) = 5.0
		end

		-- The root-level generation function
		return qs.func(terra(registers: &Regs)
			if not isinitialized then
				init()
			end
			return [genFn(nOutChannels)](registers)
		end)

	end)
end




