
-- Pack all of the user-facing nodes into one master module, so we don't
--    have to include a ton of different files to use them all.


local moduleNames =
{
	"perlin",
	"transform",
	"warp"	
}


-------------------------------------------------------------------------------


local nodes = {}

for _,modname in ipairs(moduleNames) do
	local mod = terralib.require(string.format("tex.nodes.%s", modname))
	for k,v in pairs(mod) do
		nodes[k] = v
	end
end

return nodes
