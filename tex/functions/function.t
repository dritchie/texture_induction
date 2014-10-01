local S = terralib.require("qs.lib.std")
local inherit = terralib.require("utils.inheritance")
local Registers = terralib.require("tex.registers")
local Node = terralib.require("tex.functions.node").Node
local HashMap = terralib.require("qs.lib.hashmap")
local util = terralib.require("utils.util")


-- Some utilities for pretty printing functions
local guid = global(uint, 0)
local terra makeGUID()
	var g = guid
	guid = guid + 1
	return g
end


-- A function is an abstraction around one or more primitive texture nodes
-- All functions take input coordinate and return some value.
-- Function graphs are more convenient to work with than raw dataflow node graphs,
--    especially for coordinate transformations.
local Function
Function = S.memoize(function(real, nchannels, GPU)
	assert(type(nchannels) == "number", "Function: nchannels must be a number")
	assert(type(GPU) == "boolean", "Function: GPU must be a boolean")

	local struct FunctionT(S.Object)
	{
		registers: &Registers(real, GPU)
	}
	FunctionT.metamethods.__typename = function(self)
		local platform = GPU and "GPU" or "CPU"
		return string.format("Function(%s, %d, %s)", real, nchannels, platform)
	end

	terra FunctionT:__init(registers: &Registers(real, GPU))
		self.registers = registers
	end

	terra FunctionT:__destruct() : {} end
	inherit.virtual(FunctionT, "__destruct")

	local CoordNode = Node(real, 2, GPU)
	local OutputNode = Node(real, nchannels, GPU)

	-- Expand the primitive node graph encapsulated by this function for a given
	--    coordinate source node.
	-- The resulting graph can then be interpreted/compiled.
	inherit.purevirtual(FunctionT, "expand", {&CoordNode}->{&OutputNode})

	-- Print a human-readable description of this function as an expression tree.
	-- uint argument is tab level.
	inherit.purevirtual(FunctionT, "treePrintPretty", {uint}->{})

	-- Print a human-readable description of this function as an SSA program.
	-- HashMap argument is a cache of variable ids for sub-graphs that have been printed already.
	inherit.purevirtual(FunctionT, "ssaPrintPretty", {&HashMap(&opaque,uint)}->{})

	-- Print out any aggregate-type parameters of the function
	-- (A utility for the above two printPretty methods)
	terra FunctionT:printAggParams(tablevel: uint) : {}
		-- Does nothing by default
	end
	inherit.virtual(FunctionT, "printAggParams")

	-- Create a 'default' subclass, given data about inputs and parameters
	-- inputs is a list of tables mapping names to number of channels
	-- params is a list of tables mapping names to types
	function FunctionT.makeDefaultSubtype(name, inputs, params)

		local struct FunctionSubtype(S.Object) {}
		FunctionSubtype.metamethods.__typename = function(self)
			local platform = GPU and "GPU" or "CPU"
			return string.format("%s(%s, %d, %s)", name, real, nchannels, platform)
		end
		inherit.dynamicExtend(FunctionT, FunctionSubtype)

		FunctionSubtype.CoordNode = CoordNode
		FunctionSubtype.OutputNode = OutputNode

		local inputsyms = terralib.newlist()
		local paramsyms = terralib.newlist()
		for _,it in ipairs(inputs) do
			for name,nchann in pairs(it) do
				local typ = &Function(real, nchann, GPU)
				FunctionSubtype.entries:insert({field=name, type=typ})
				inputsyms:insert(symbol(typ, name))
			end
		end
		for _,pt in ipairs(params) do
			for name,typ in pairs(pt) do
				FunctionSubtype.entries:insert({field=name, type=typ})
				paramsyms:insert(symbol(typ, name))
			end
		end

		terra FunctionSubtype:__init(registers: &Registers(real, GPU), [inputsyms], [paramsyms])
			FunctionT.__init(self, registers)
			escape
				for _,s in ipairs(inputsyms) do
					emit quote self.[s.displayname] = s end
				end
				for _,s in ipairs(paramsyms) do
					emit quote S.copy(self.[s.displayname], s) end
				end
			end
		end

		terra FunctionSubtype:__destruct() : {}
			escape
				for _,s in ipairs(inputsyms) do
					emit quote
						if self.[s.displayname] ~= nil then
							self.[s.displayname]:delete()
						end
						self.[s.displayname] = nil
					end
				end
			end
		end
		inherit.virtual(FunctionSubtype, "__destruct")

		-- Print the function name, the values of its parameters, and then recursively print
		--    its inputs
		terra FunctionSubtype:treePrintPretty(tablevel: uint) : {}
			S.printf("%s(\n", name)
			escape
				-- Print params
				for _,ps in ipairs(paramsyms) do
					-- (We only handle int and float types directly)
					if ps.type:isarithmetic() then
						local fmtstr = ps.type:isfloat() and "%s: %g\n" or "%s: %d\n"
						emit quote
							util.printTabs(tablevel+1)
							S.printf(fmtstr, [ps.displayname], self.[ps.displayname])
						end
					end
				end
				-- Print other params, if supported
				emit quote self:printAggParams(tablevel+1) end
				-- Recursively print inputs
				for _,is in ipairs(inputsyms) do
					emit quote
						util.printTabs(tablevel+1)
						S.printf("%s: ", [is.displayname])
						self.[is.displayname]:treePrintPretty(tablevel + 1)
					end
				end
			end
			util.printTabs(tablevel)
			S.printf(")\n")
		end
		inherit.virtual(FunctionSubtype, "treePrintPretty")

		-- Recursively print inputs (thus causing any necessary prerequisite variable assignments
		--    to be printed), then print the function itself.
		terra FunctionSubtype:ssaPrintPretty(addrToId: &HashMap(&opaque, uint)) : {}
			escape
				-- Recursively print inputs
				for _,is in ipairs(inputsyms) do
					emit quote self.[is.displayname]:ssaPrintPretty(addrToId) end
				end
				-- Print assignment for this function call
				emit quote
					var myId = makeGUID()
					addrToId:put([&opaque](self), myId)
					S.printf("$%u = %s(\n", myId, name)
				end
				-- Print params
				for _,ps in ipairs(paramsyms) do
					if ps.type:isarithmetic() then
						local fmtstr = ps.type:isfloat() and "%s: %g\n" or "%s: %d\n"
						emit quote
							util.printTabs(1)
							S.printf(fmtstr, [ps.displayname], self.[ps.displayname])
						end
					end
				end
				-- Print other params, if supported
				emit quote self:printAggParams(1) end
				-- Print inputs by looking up into the addrToId cache
				for _,is in ipairs(inputsyms) do
					emit quote
						var idPtr = addrToId:getPointer([&opaque](self.[is.displayname]))
						S.assert(idPtr ~= nil)
						util.printTabs(1)
						S.printf("%s: $%u\n", [is.displayname], @idPtr)
					end
				end
			end
			S.printf(")\n")
		end
		inherit.virtual(FunctionSubtype, "ssaPrintPretty")

		return FunctionSubtype
	end


	-- TODO: Support compilation directly from Functions (to avoid generating
	--    duplicate subgraphs in some cases)???

	return FunctionT

end)


return Function



