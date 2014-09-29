local S = terralib.require("qs.lib.std")
local inherit = terralib.require("utils.inheritance")
local Registers = terralib.require("tex.registers")
local Node = terralib.require("tex.functions.node").Node


-- A function is an abstraction around one or more primitive texture nodes
-- All functions take input coordinate and return some value.
-- Function graphs are more convenient to work with than raw dataflow node graphs,
--    especially for coordinate transformations.
local Function
Function = S.memoize(function(real, nchannels, GPU)

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
	-- inherit.virtual(FunctionT, "__destruct")

	local CoordNode = Node(real, 2, GPU)
	local OutputNode = Node(real, nchannels, GPU)

	-- Expand the primitive node graph encapsulated by this function for a given
	--    coordinate source node.
	-- The resulting graph can then be interpreted/compiled.
	inherit.purevirtual(FunctionT, "expand", {&CoordNode}->{&OutputNode})


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

		return FunctionSubtype
	end


	-- TODO: Support compilation directly from Functions (to avoid generating
	--    duplicate subgraphs in some cases)???

	return FunctionT

end)


return Function



