local S = terralib.require("qs.lib.std")
local image = terralib.require("utils.image")
local CUDAImage = terralib.require("utils.cuimage")


-- A reusable pool of image data (to avoid allocating more images than necessary)
local ImagePool = S.memoize(function(real, nchannels, GPU)

	local Image
	if GPU then
		Image = CUDAImage(real, nchannels)
	else
		Image = image.Image(real, nchannels)
	end

	local struct ImagePool(S.Object)
	{
		images: S.Vector(&Image),
		freeStack: S.Vector(&Image)
	}

	terra ImagePool:__destruct()
		for i=0,self.images:size() do
			self.images(i):delete()
		end
	end

	-- Fetch an image of the requested size
	-- (Internally, we don't record sizes--this will just
	--    fetch the first available image and resize it if needed.
	--    A lot of resizing kind of defeats the purpose of having
	--    this class in the first place, so the expected use case is
	--    one where the size is pretty much constant.
	terra ImagePool:fetch(xres: uint, yres: uint)
		-- If the freeStack is empty, then we have no free images available
		-- In this case, allocate a new image
		if self.freeStack:size() == 0 then
			var newimg = Image.alloc():init(xres, yres)
			self.images:insert(newimg)
			self.freeStack:insert(newimg)
		end
		-- Pop and return the top of the freeStack, resizing it if need be
		var retimg = self.freeStack:remove()
		retimg:resize(xres, yres)
		return retimg
	end

	-- Release an image back to the pool for another client to use.
	-- It's assumed that this image came from the pool to begin with;
	--    calling release with an image allocated elsewhere leads to
	--    undefined behavior.
	terra ImagePool:release(img: &Image)
		self.freeStack:insert(img)
	end

	return ImagePool

end)


return ImagePool