local S = terralib.require("qs.lib.std")
local qs = terralib.require("qs")
local fns = terralib.require("tex.functions.functions")
local Function = terralib.require("tex.functions.function")
local Registers = terralib.require("tex.registers")
local randTables = terralib.require("tex.randTables")
local Vec = terralib.require("utils.linalg.vec")
local Mat = terralib.require("utils.linalg.mat")
local mathlib = terralib.require("utils.mathlib")


-- Incredibly simple grammar for generating texture programs.
-- Starts at the output function.
-- Generates each input independently.
-- Uses the same rules no matter what type of function it's currently expanding.
return function(nOutChannels, GPU)
	assert(nOutChannels == 1 or nOutChannels == 4,
		"grammar: nOutChannels must be either 1 (grayscale) or 4 (color)")
	local mlib = mathlib(GPU)
	return qs.module(function()

		-- Typedefs
		local Regs = Registers(qs.real, GPU)
		local GrayscaleFn = Function(qs.real, 1, GPU)
		local ColorFn = Function(qs.real, 4, GPU)
		local GrayscaleFnGenerator = {&Regs}->{&GrayscaleFn}
		local ColorFnGenerator = {&Regs}->{&ColorFn}

		-- Globals
		local grayscaleProbs = global(S.Vector(qs.real))
		local grayscaleGens = global(S.Vector(GrayscaleFnGenerator))
		local colorProbs = global(S.Vector(qs.real))
		local colorGens = global(S.Vector(ColorFnGenerator))


		-- The master generator function
		local genFn = S.memoize(function(nchannels)
			local probs = nchannels == 4 and colorProbs or grayscaleProbs
			local gens = nchannels == 4 and colorGens or grayscaleGens
			return qs.func(terra(registers: &Regs)
				var which = qs.categorical(&probs)
				return gens(which)(registers)
			end)
		end)

		
		-----------------------------------------------------
		-- Random generators for different function types  --
		-----------------------------------------------------

		local genPerlin = S.memoize(function()
			local gradients = randTables.const_gradients(qs.real, GPU)
			return qs.func(terra(registers: &Regs)
				var frequency = qs.gammamv(1.0, 0.5, {struc=false})
				var lacunarity = qs.gammamv(2.0, 1.0, {struc=false})
				var persistence = qs.betamv(0.5, 0.05, {struc=false})
				-- TODO: Should these be considered structural, or no?
				var startOctave = qs.poisson(1, struc=false)
				var octaves = qs.poisson(5, {struc=false})
				return [fns.Perlin(qs.real, GPU)].alloc():init(registers, gradients,
					frequency, lacunarity, persistence, startOctave, octaves)
			end)
		end)

		local genTransform = S.memoize(function(nchannels)
			local Mat3 = Mat(qs.real, 3, 3, GPU)
			return qs.func(terra(registers: &Regs)
				var input = [genFn(nchannels)](registers)
				-- TODO: Should we allow negative scales (i.e. reflections) as well?
				var scalex = mlib.exp(qs.gaussian(0.0, 1.0, {struc=false}))
				var scaley = mlib.exp(qs.gaussian(0.0, 1.0, {struc=false}))
				var ang = qs.gaussian(0.0, [math.pi/4.0], struc=false)
				var xform = Mat3.rotate(ang) * Mat3.scale(scalex, scaley)
				return [fns.Transform(qs.real, nchannels, GPU)].alloc():init(registers, input, xform)
			end)
		end)

		local genDecolorize = S.memoize(function()
			return qs.func(terra(registers: &Regs)
				var input = [genFn(4)](registers)
				return [fns.Decolorize(qs.real, GPU)].alloc():init(registers, input)
			end)
		end)

		local genWarp = S.memoize(function(nchannels)
			return qs.func(terra(registers: &Regs)
				var input = [genFn(nchannels)](registers)
				var warpfield = [genFn(1)](registers)
				var strength = qs.gammamv(0.05, 0.2, {struc=false})
				return [fns.Warp(qs.real, nchannels, GPU)].alloc():init(registers, input, warpfield, strength)
			end)
		end)

		local genMask = S.memoize(function(nchannels)
			return qs.func(terra(registers: &Regs)
				var bot = [genFn(nchannels)](registers)
				var top = [genFn(nchannels)](registers)
				var mask = [genFn(1)](registers)
				return [fns.Mask(qs.real, nchannels, GPU)].alloc():init(registers, bot, top, mask)
			end)
		end)

		local genColorize = S.memoize(function()
			local RGBAColor = Vec(qs.real, 4, GPU)
			-- Uniform prior over num points
			-- TODO: Better prior on num points
			local MAX_NUM_GRAD_POINTS = fns.Colorize(qs.real, GPU).MAX_NUM_GRAD_POINTS
			local nPointsProbs = global(qs.real[MAX_NUM_GRAD_POINTS])
			for i=1,MAX_NUM_GRAD_POINTS do
				nPointsProbs:get()[i-1] = 1.0/MAX_NUM_GRAD_POINTS
			end
			return qs.func(terra(registers: &Regs)
				var input = [genFn(1)](registers)
				var knots = [S.Vector(qs.real)].salloc():init()
				var colors = [S.Vector(RGBAColor)].salloc():init()
				var npoints = qs.categorical(nPointsProbs) + 1
				var knotSpaceLeft = qs.real(1.0)
				for i=0,npoints do
					-- Generate knots by stick-breaking
					-- (Take care to ensure that knots will remain correct under any perturbation)
					var knot = qs.uniform(0.0, 1.0, {struc=false}) * knotSpaceLeft
					knotSpaceLeft = knotSpaceLeft - knot
					knots:insert(knot)
					-- Generate colors uniformly at random
					-- TODO: Make color prior encourage more smooth change / continuity
					colors:insert(RGBAColor.create(qs.uniform(0.0, 1.0, {struc=false}),
												   qs.uniform(0.0, 1.0, {struc=false}),
												   qs.uniform(0.0, 1.0, {struc=false}),
												   qs.uniform(0.0, 1.0, {struc=false})))
				end
			end)
		end)

		local genBlend = S.memoize(function()
			return qs.func(terra(registers: &Regs)
				var bot = [genFn(4)](registers)
				var top = [genFn(4)](registers)
				var opacity = qs.uniform(0.0, 1.0, {struc=false})
				return [fns.Blend(qs.real, GPU)].alloc():init(registers, bot, top, opacity)
			end)
		end)


		-- Initialize the global prob/gen lists
		local terra init()
			grayscaleProbs:init()
			grayscaleGens:init()
			colorProbs:init()
			colorGens:init()

			-- Fill in the global probs/gens lists
			-- (Just use uniform probabilities for now)
			grayscaleGens:insert([genPerlin()])
			grayscaleGens:insert([genTransform(1)])
			grayscaleGens:insert([genDecolorize()])
			grayscaleGens:insert([genWarp(1)])
			grayscaleGens:insert([genMask(1)])
			for i=0,grayscaleGens:size() do grayscaleProbs:insert(1.0/grayscaleGens:size()) end
			colorGens:insert([genTransform(4)])
			colorGens:insert([genColorize()])
			colorGens:insert([genWarp(4)])
			colorGens:insert([genBlend()])
			colorGens:insert([genMask(4)])
			for i=0,colorGens:size() do colorProbs:insert(1.0/colorGens:size()) end
		end
		init()


		-- The root-level generation function
		return qs.func(terra(registers: &Regs)
			return [genFn(nOutChannels)](registers)
		end)

	end)
end




