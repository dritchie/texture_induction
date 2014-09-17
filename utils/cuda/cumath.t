
local libdevice = terralib.linklibrary("lib/libdevice.bc")


local cumath = {}


local function addfn(fname, nargs)
	nargs = nargs or 1
	-- Add double function
	local ptype = terralib.newlist()
	for i=1,nargs do ptype:insert(double) end
	cumath[fname] = terralib.externfunction(string.format("__nv_%s", fname), ptype->double)
	-- Add float function
	local ptype = terralib.newlist()
	for i=1,nargs do ptype:insert(float) end
	cumath[fname] = terralib.externfunction(string.format("__nv_%sf", fname), ptype->float)
end



addfn("acos")
addfn("acosh")
addfn("asin")
addfn("asinh")
addfn("atan")
addfn("atan2", 2)
addfn("ceil")
addfn("cos")
addfn("cosh")
addfn("exp")
addfn("fabs")
addfn("floor")
addfn("fmax", 2)
addfn("fmin", 2)
addfn("log")
addfn("log10")
addfn("pow", 2)
addfn("round")
addfn("sin")
addfn("sinh")
addfn("sqrt")
addfn("tan")
addfn("tanh")



return cumath


