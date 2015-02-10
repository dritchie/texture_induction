local S = require("qs.lib.std")
local Vec = require("utils.linalg.vec")
local randTables = require("tex.randTables")
local mathlib = require("utils.mathlib")

local GradientTable = randTables.GradientTable


-- Much of this implementation of basic noise primitives is taken from
--    libnoise (http://libnoise.sourceforge.net/index.html)


-- Constants for index shuffling
local X_NOISE_GEN = 1619
local Y_NOISE_GEN = 31337
local SHIFT_NOISE_GEN = 8
local SEED_NOISE_GEN = 1013


-- Interpolation
local lerp = macro(function(lo, hi, t)
	return `(1.0-t)*lo + t*hi
end)
local ease = macro(function(t)
	return `(t * t * (3.0 - 2.0 * t))
end)



-- Helper: Compute the gradient noise value an integer grid location,
--    given the continuous location where noise will eventually be evaluated.
local gradientNoise = S.memoize(function(real, GPU)
	return terra(fx: real, fy: real, x: int, y: int, seed: int, grads: GradientTable(real, GPU))
		-- Lookup into the gradient table using a random permutation of x,y
		var index = (
			X_NOISE_GEN * x +
			Y_NOISE_GEN * y +
			SEED_NOISE_GEN * seed
		) and 0xffffffff	-- Not sure why this is necessary...?
		index = index ^ (index >> SHIFT_NOISE_GEN)
		-- Ensure all indices are in range
		index = index and [randTables.GRADIENT_TABLE_SIZE - 1]
		var g = grads[index]

		-- Take the dot product of this gradient with the vector between (fx,fy)
		--    and (x,y)
		-- Also rescale to be in the range (-1, 1)
		var v = @[Vec(real,2,GPU)].salloc():init(fx-x, fy-y)
		return v:dot(g) * [1.0/math.sqrt(2)]
	end
end)

-- Compute coherent gradient noise at a point
local gradientCoherentNoise = S.memoize(function(real, GPU)
	local gradNoise = gradientNoise(real, GPU)
	local mlib = mathlib(GPU)
	return terra(x: real, y: real, seed: int, grads: GradientTable(real, GPU))
		-- Bound the input point within an integer grid cell
		var x0 = int(mlib.floor(x))
		var x1 = x0 + 1
		var y0 = int(mlib.floor(y))
		var y1 = y0 + 1

		-- Cubic interpolate the distance from the origin of the cell
		--    to the input point
		var xs = ease(x - x0)
		var ys = ease(y - y0)

		-- Calculate noise values at each grid vertex, then bilinear
		--    interpolate to get final noise value
		-- S.printf("--------------\n")
		var n00 = gradNoise(x, y, x0, y0, seed, grads)
		var n01 = gradNoise(x, y, x0, y1, seed, grads)
		var n0 = lerp(n00, n01, ys)
		var n10 = gradNoise(x, y, x1, y0, seed, grads)
		var n11 = gradNoise(x, y, x1, y1, seed, grads)
		var n1 = lerp(n10, n11, ys)
		return lerp(n0, n1, xs)
	end
end)



return
{
	gradientCoherent = gradientCoherentNoise
}




