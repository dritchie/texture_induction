local S = require("qs.lib.std")

local C = terralib.includecstring [[
#include "stdio.h"
]]

-- Pretty much just like stl::auto_ptr

local struct RefCount(S.Object)
{
	count: uint
}

terra RefCount:__init()
	self.count = 1
end

terra RefCount:retain()
	self.count = self.count + 1
end

terra RefCount:release()
	-- Cannot release on a refcount with zero references
	S.assert(self.count > 0)
	self.count = self.count - 1
end

terra RefCount:empty()
	return self.count == 0
end



local AutoPtr = S.memoize(function(T)
	local struct AutoPtrT(S.Object)
	{
		ptr: &T,
		refCount: &RefCount
	}

	AutoPtrT.metamethods.__typename = function(self)
		return string.format("AutoPtr(%s)", tostring(T))
	end

	terra AutoPtrT:__init()
		self.ptr = nil
		self.refCount = nil
	end

	terra AutoPtrT:__init(ptr: &T)
		-- S.printf("create %p\n", ptr)
		self.ptr = ptr
		self.refCount = RefCount.alloc():init()
	end

	terra AutoPtrT:__copy(other: &AutoPtrT)
		-- S.printf("retain %p\n", other.ptr)
		self.ptr = other.ptr
		self.refCount = other.refCount
		self.refCount:retain()
	end

	terra AutoPtrT:__destruct()
		if self.ptr ~= nil and self.refCount ~= nil then
			-- S.printf("release %p\n", self.ptr)
			self.refCount:release()
			if self.refCount:empty() then
				self.refCount:delete()
				self.ptr:delete()
			end
		end
	end

	AutoPtrT.metamethods.__entrymissing = macro(function(fieldname, self)
		return `self.ptr.[fieldname]
	end)
	
	-- I use this more complicated behavior, rather than just using __methodmissing,
	--    because I want AutoPtrT:getmethod to still return nil exactly when T:getmethod
	--    would return nil.
	AutoPtrT.metamethods.__getmethod = function(self, methodname)
		-- If AutoPtrT has the method (i.e. is it __construct, __destruct, __copy),
		--    then just return that
		local mymethod = self.methods[methodname]
		if mymethod then return mymethod end
		-- Otherwise, if T has it, then return a macro that will invoke T's
		--    method on the .ptr member
		local tmethod = T:getmethod(methodname)
		if tmethod then
			return macro(function(self, ...)
				local args = {...}
				return `[tmethod](self.ptr, [args])
			end)
		end
		-- Otherwise, return nil
		return nil
	end

	return AutoPtrT
end)


-- Convience macro to create a new auto pointer without explicitly specifying the type
-- It'll figure out the type from the type of the argument
local create = macro(function(ptr)
	local pT = ptr:gettype()
	assert(pT:ispointertostruct(),
		"Can only create an auto pointer from a pointer to a struct.")
	local T = pT.type
	return quote
		var aptr : AutoPtr(T)
		aptr:init(ptr)
	in
		aptr
	end
end)



return
{
	AutoPtr = AutoPtr,
	create = create
}






