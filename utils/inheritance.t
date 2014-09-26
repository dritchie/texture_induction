local S = terralib.require("qs.lib.std")


-- Really simple single inheritance

local Inheritance = {}

-- metadata for class system
local metadata = {}

local function issubclass(child,parent)
	if child == parent then
		return true
	else
		local par = metadata[child].parent
		return par and issubclass(par,parent)
	end
end
Inheritance.issubclass = issubclass

local function setParent(child, parent)
	local md = metadata[child]
	if md then
		if md.parent then
			error(string.format("'%s' already inherits from some type -- multiple inheritance not allowed.", child))
		end
		md.parent = parent
	else
		metadata[child] = {parent = parent}
	end
end

local function castoperator(from, to, exp)
	if from:ispointer() and to:ispointer() and issubclass(from.type, to.type) then
		return `[to](exp)
	else
		error(string.format("'%s' does not inherit from '%s'", from.type, to.type))
	end
end

local function lookupParentStaticMethod(class, methodname)
	local cls = class
	while cls ~= nil do
		if cls.methods[methodname] ~= nil then
			return cls.methods[methodname]
		else
			if metadata[cls] and metadata[cls].parent then
				cls = metadata[cls].parent
			else
				cls = nil
			end
		end
	end
	return nil
end

local function copyparentlayoutStatic(class)
	local parent = metadata[class].parent
	for i,e in ipairs(parent.entries) do table.insert(class.entries, i, e) end
	return class.entries
end

local function addstaticmetamethods(class)
	class.metamethods.__cast = castoperator
	class.metamethods.__getentries = copyparentlayoutStatic
	class.metamethods.__getmethod = lookupParentStaticMethod
end


-- child inherits data layout and method table from parent
function Inheritance.staticExtend(parent, child)
	setParent(child, parent)
	addstaticmetamethods(child)
end


------------------------------------------

-- Create the function which will initialize the __vtable field
--    in each struct instance.
-- Then, wrap the struct's "init" method with a macro that will invoke
--    this function.
-- Throw an error if the class does not have an "init" method.
local function makeinitvtable(class)
	local md = metadata[class]
	-- global, because otherwise it would be GC'ed.
	md.vtable = global(md.vtabletype)
	-- Add the vtable initializer (or augment an existing one)
	-- (e.g. a virtual template could have already set up this method)
	local oldinitvtable = class.methods.__initvtable
	class.methods.__initvtable = terra(self: &class)
		[oldinitvtable and (quote oldinitvtable(self) end) or (quote end)]
		self.__vtable = &md.vtable
	end
	if not class.methods.init then
		error(string.format("Struct %s uses virtual methods but has no 'init' method; vtable cannot be initialized.",
			tostring(class)))
	end
	local oldinit = class.methods.init
	class.methods.init = macro(function(self, ...)
		local args = {...}
		return quote
			var s = &self
			(@s):__initvtable()
		in
			oldinit(@s, [args])
		end
	end)
end

-- Finalize the vtable after the class has been compiled
local function finalizevtable(class)
	local md = metadata[class]
	local vtbl = md.vtable:get()
	for methodname,impl in pairs(md.methodimpl) do
		impl:compile(function()
			vtbl[methodname] = impl:getpointer()  
		end)
	end
end

-- Create a 'stub' method which refers to the method of the same
-- name in the class's vtable
local function createstub(methodname,typ)
	local symbols = typ.parameters:map(symbol)
	local obj = symbols[1]
	local terra wrapper([symbols]) : typ.returntype
		return obj.__vtable.[methodname]([symbols])
	end
	return wrapper
end

local function getdefinitionandtype(impl)
	if #impl:getdefinitions() ~= 1 then
			error(string.format("Overloaded function '%s' cannot be virtual.", method.name))
		end
	local impldef = impl:getdefinitions()[1]
	local success, typ = impldef:peektype()
	if not success then
		error(string.format("virtual method '%s' must have explicit return type", impl.name))
	end
	return impldef,typ
end

-- In order for the standard library destructor behavior to work
--    correctly with subtype polymorphim (i.e. when an object's dynamic
--    type can differ from its static type), the default destruct behavior
--    (which is to destruct all of the object members) must be virtual.
-- Notice that the __destructmembers function does almost the same thing as
--    'generatedtor' in lib.std, but it wraps this logic in a virtual function.
local function makeVirtualDestructMembers(class)
	terra class:__destructmembers() : {}
		-- S.printf("virtual destruct members\n")
		escape
		    for i,e in ipairs(class.entries) do
		        if e.field and e.field ~= "__vtable" then --not a union, not a vtable
		        	-- emit `S.printf("\t%s : %s\n", [e.field], [tostring(e.type)])
		            emit `S.rundestructor(self.[e.field])
		        end
		    end
		end
	end
	Inheritance.virtual(class, "__destructmembers")
end

-- Finalize the layout of the struct
local function finalizeStructLayoutDynamic(class)
	local md = metadata[class]

	-- Start up the vtable data
	struct md.vtabletype {}
	md.methodimpl = {}

	-- Create __vtable field
	class.entries:insert(1, { field = "__vtable", type = &md.vtabletype})

	-- Copy data from parent
	local parent = md.parent
	if parent then
		-- Must do this to make sure the parent's layout has been finalized first
		parent:getentries()
		-- Static members (except the __vtable field)
		for i=2,#parent.entries do
			class.entries:insert(i, parent.entries[i])
		end
		-- vtable entries
		local pmd = metadata[parent]
		for i,m in ipairs(pmd.vtabletype.entries) do
			md.vtabletype.entries:insert(m)
			md.methodimpl[m.field] = pmd.methodimpl[m.field]
		end
	end

	-- Make the virtual default destructor for this class
	-- This MUST happen *after* class.entries has been populated with parent entries,
	--    but before the class's own methods are copies into vtable staging (meaning,
	--    this space is the only place it can go).
	makeVirtualDestructMembers(class)

	-- Copy all my virtual methods into the vtable staging area
	for methodname, impl in pairs(class.methods) do
		if md.vmethods and md.vmethods[methodname] then
			local def, typ = getdefinitionandtype(impl)
			if md.methodimpl[methodname] == nil then
				md.vtabletype.entries:insert({field = methodname, type = &typ})
			end
			md.methodimpl[methodname] = def
		end
	end

	-- Create method stubs (overwriting any methods marked virtual)
	for methodname, impl in pairs(md.methodimpl) do
		local _,typ = impl:peektype()
		class.methods[methodname] = createstub(methodname, typ)
	end

	-- Make the class's "init" method initialize the __vtable field
	makeinitvtable(class)

	return class.entries
end


-- Add metamethods necessary for dynamic dispatch
local function adddynamicmetamethods(class)
	class.metamethods.__cast = castoperator
	class.metamethods.__staticinitialize = finalizevtable
	class.metamethods.__getentries = finalizeStructLayoutDynamic
	class.metamethods.__getmethod = lookupParentStaticMethod
end


-- Ensure that a struct is equipped for dynamic dispatch
-- (i.e. has a vtable, has the requisite metamethods)
local function ensuredynamic(class)
	if not metadata[class] then
		metadata[class] = {}
	end
	adddynamicmetamethods(class)
end

-- Mark a method as virtual
function Inheritance.virtual(class, methodname)
	ensuredynamic(class)
	local md = metadata[class]
	if not md.vmethods then
		md.vmethods = {}
	end
	md.vmethods[methodname] = true
end

-- Create a 'stub' method of type typ which throws a 
--    'not implemented' error.
local function createunimplementedstub(class, methodname, typ)
	local symbols = typ.parameters:map(symbol)
	local obj = symbols[1]
	local terra wrapper([symbols]) : typ.returntype
		S.printf("Pure virtual function '%s' not implemented in class '%s'\n", methodname, [tostring(class)])
		S.assert(false)
	end
	return wrapper
end

-- Declare a pure virtual function (no implementation)
function Inheritance.purevirtual(class, methodname, typ)
	-- Expand the type to include the pointer to self
	local params = terralib.newlist()
	for _,p in ipairs(typ.type.parameters) do params:insert(p) end
	local returntype = typ.type.returntype
	table.insert(params, 1, &class)
	typ = terralib.types.funcpointer(params, returntype)
	-- Add an 'unimplemented' method with this name to the class
	class.methods[methodname] = createunimplementedstub(class, methodname, typ.type)
	-- Now do all the stuff we usually do for virtual methods.
	Inheritance.virtual(class, methodname)
end


-- child inherits data layout and method table from parent
-- child also inherits vtable from parent
function Inheritance.dynamicExtend(parent, child)
	ensuredynamic(parent)
	ensuredynamic(child)
	setParent(child, parent)
end

function Inheritance.isInstanceOf(T)

	-- Lua callback that first finds the dynamic type associated with
	--    a vtable, then checks if that type is a descendant of T
	local function isDynamicSubtype(vptr)
		local dyntyp = nil
		for t,md in pairs(metadata) do
			if md.vtable and md.vtable:getpointer() == vptr then
				dyntyp = t
				break
			end
		end
		if dyntyp == nil then return false end
		while dyntyp do
			if dyntyp == T then return true end
			dyntyp = metadata[dyntyp] and metadata[dyntyp].parent
		end
		return false
	end
	isDynamicSubtype = terralib.cast({&opaque}->{bool}, isDynamicSubtype)

	return macro(function(inst)
		local t = inst:gettype()
		if t:ispointertostruct() then t = t.type end
		-- First check: is t a subtype of T?
		if issubclass(t, T) then return true end
		-- Otherwise, we need to compare vtable pointers
		-- (Are these getentries() calls safe???)
		T:getentries()
		t:getentries()
		-- Not possible if t doesn't have a vtable
		if not (metadata[t] and metadata[t].vtable) then return false end

		return `isDynamicSubtype([&opaque](inst.__vtable))
	end)
end


----------------------------------------------------------------------------


-- -- TEST

-- local struct A(S.Object) { x : int }

-- terra A:__init(x: int)
-- 	S.printf("A init\n")
-- 	self.x = x
-- end

-- terra A:__destruct() : {} S.printf("A destruct\n") end
-- Inheritance.virtual(A, "__destruct")

-- terra A:report() : {} S.printf("A report\n") end
-- Inheritance.virtual(A, "report")

-- local struct B(S.Object) { vec : S.Vector(int) }
-- Inheritance.dynamicExtend(A, B)

-- terra B:__init(x: int)
-- 	S.printf("B init\n")
-- 	self:initmembers()
-- 	self.x = x
-- end

-- terra B:__destruct() : {} S.printf("B destruct\n") end
-- Inheritance.virtual(B, "__destruct")

-- terra B:report() : {} S.printf("B report\n") end
-- Inheritance.virtual(B, "report")

-- local terra test()
-- 	var a : &A = A.alloc():init(0)
-- 	var b : &B = B.alloc():init(1)
-- 	var ab : &A = B.alloc():init(2)

-- 	a:report()
-- 	b:report()
-- 	ab:report()

-- 	a:delete()
-- 	b:delete()
-- 	ab:delete()
-- end
-- -- test:printpretty()
-- test()


----------------------------------------------------------------------------


return Inheritance






