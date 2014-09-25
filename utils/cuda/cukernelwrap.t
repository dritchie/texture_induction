

-- Wrap a Terra function such that it takes no struct or array arguments by value.
-- All occurrences of struct(or array)-by-value are unpacked before being passed in
--    and then re-packed once inside the function.
return function(terrafn, verbose)
	-- Workaround for NVPTX issues with generating code for structs passed by value.
	terrafn:setinlined(true)
	terrafn:emitllvm()	-- To guarantee we have a type
	local succ, T = terrafn:peektype()
	assert(succ)
	assert(T.returntype:isunit(), "cukernelwrap: kernel must not return anything.")

	-- All the work is done here
	local function recurse(exprs, types)
		local kernelSyms = terralib.newlist()
		local unpackExprs = terralib.newlist()
		local repackExprs = terralib.newlist()
		for i,exp in ipairs(exprs) do
			local typ = types[i]
			-- Base case: primitive/pointer types can be passed in directly
			if typ:ispointer() or typ:isprimitive() then
				local sym = symbol(typ)
				kernelSyms:insert(sym)
				unpackExprs:insert(exp)
				repackExprs:insert(sym)
			-- Recursive cases
			elseif typ:isstruct() then
				local recExprs = typ.entries:map(function(e) return `exp.[e.field] end)
				local recTypes = typ.entries:map(function(e) return e.type end)
				local recKernelSyms, recUnpackExprs, recRepackExprs = recurse(recExprs, recTypes)
				kernelSyms:insertall(recKernelSyms)
				unpackExprs:insertall(recUnpackExprs)
				repackExprs:insert(`typ { [recRepackExprs] })
			elseif typ:isarray() then
				local recExprs = terralib.newlist()
				local recTypes = terralib.newlist()
				for i=1,typ.N do
					recExprs:insert(`exp[ [i-1] ])
					recTypes:insert(typ.type)
				end
				local recKernelSyms, recUnpackExprs, recRepackExprs = recurse(recExprs, recTypes)
				kernelSyms:insertall(recKernelSyms)
				unpackExprs:insertall(recUnpackExprs)
				repackExprs:insert(`array([recRepackExprs]))
			else
				error(string.format("cukernelwrap: type %s not a primitive, pointer, struct, or array. Impossible?",
					tostring(typ)))
			end
		end
		return kernelSyms, unpackExprs, repackExprs
	end

	local outersyms = T.parameters:map(function(t) return symbol(t) end)
	local kernelSyms, unpackExprs, repackExprs = recurse(outersyms, T.parameters)

	-- The actual kernel takes unpacked args, re-packs them, then calls the original
	--    function passed in by the user.
	local terra kernel([kernelSyms]) : {}
		terrafn([repackExprs])
	end
	local K = terralib.cudacompile({kernel=kernel}, verbose)

	-- We return a wrapper around the kernel that takes the original arguments, unpacks
	--    them, then calls the kernel.
	return terra(kernelparams: &terralib.CUDAParams, [outersyms]) : {}
		K.kernel(kernelparams, [unpackExprs])
	end
end