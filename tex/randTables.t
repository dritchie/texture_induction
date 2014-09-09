local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local distrib = terralib.require("qs.distrib")


-- Should be a power of 2
local gradientTableSize = 256

local GradientTable = S.memoize(function(real)
	return Vec(real, 2)[gradientTableSize]
end)


-- Gradient noise is driven by a table of random gradient vectors. We may experiment with treating
--    this table as a random variable, but the typical implementation has a hardcoded constant table.
--    This is our version of that 'constant table' (a 256-element table)
local const_gradients = S.memoize(function(real)
	local Vec2 = Vec(real, 2)
	local gradients = global(GradientTable)
	local terra fillgradients()
		-- Generate a bunch of uniformly-distributed random vectors.
		-- These probably won't be very nice (i.e. something low-discrepancy would be better),
		--    but this is essentially the method we'll be stuck with if we end up making these
		--    random variables
		escape
			for i=1,gradientTableSize do
				emit quote
					var ang = [distrib.uniform(real)].sample(0.0, [2*math.pi])
					gradients[ [i-1] ] = Vec2.fromPolar(1.0, ang)
				end
			end
		end
	end
	fillgradients()
	return gradients
end)



return
{
	GradientTable = GradientTable,
	const_gradients = const_gradients
}



