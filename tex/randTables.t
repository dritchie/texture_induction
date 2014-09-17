local S = terralib.require("qs.lib.std")
local Vec = terralib.require("utils.linalg.vec")
local distrib = terralib.require("qs.distrib")
local curt = terralib.require("utils.cuda.curt")


-- Should be a power of 2
local GRADIENT_TABLE_SIZE = 256

local GradientTable = S.memoize(function(real, GPU)
	return &Vec(real, 2, GPU)
end)


-- Gradient noise is driven by a table of random gradient vectors. We may experiment with treating
--    this table as a random variable, but the typical implementation has a hardcoded constant table.
--    This is our version of that 'constant table' (a 256-element table)
local const_gradients = S.memoize(function(real, GPU)
	local Vec2 = Vec(real, 2, false)
	local gradients = global(GradientTable(real, GPU))
	local terra fillgradients()
		-- Generate a bunch of uniformly-distributed random vectors.
		var grads = [GradientTable(real, false)](S.malloc(GRADIENT_TABLE_SIZE*sizeof(Vec2)))
		for i=0,GRADIENT_TABLE_SIZE do
			-- TODO: Something low-disrepancy would be better...
			var ang = [distrib.uniform(real)].sample(0.0, [2*math.pi])
			grads[i] = Vec2.fromPolar(1.0, ang)
		end
		escape
			if GPU then
				emit quote curt.cudaMemcpy(gradients, grads, GRADIENT_TABLE_SIZE*sizeof(Vec2)) end
			else
				emit quote gradients = grads end
			end
		end
	end
	fillgradients()
	return gradients
end)



return
{
	GRADIENT_TABLE_SIZE = GRADIENT_TABLE_SIZE,
	GradientTable = GradientTable,
	const_gradients = const_gradients
}



