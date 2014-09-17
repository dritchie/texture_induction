local S = terralib.require("qs.lib.std")
local curt = terralib.includec("cuda_runtime.h")
local image = terralib.require("utils.image")
local Vec = terralib.require("utils.linalg.vec")


-- Image data stored in GPU memory
local CUDAImage = S.memoize(function(real, nchannels)

	local Color = Vec(real, nchannels, true)

	local struct CUDAImage(S.Object)
	{
		data: &Color,
		width: uint,
		height: uint,
		pitch: uint
	}

	terra CUDAImage:__init() : {}
		self.width = 0
		self.height = 0
		self.pitch = 0
		self.data = nil
	end	

	terra CUDAImage:__init(width: uint, height: uint) : {}
		self:__init()
		self.width = width
		self.height = height
		if width*height > 0 then
			var s = sizeof(Color)
			curt.cudaMallocPitch([&&opaque](&self.data), &self.pitch, s*width, s*height)
		end
	end

	-- Copy from a CPU image
	terra CUDAImage:__init(img: &image.Image(real, nchannels)) : {}
		self:__init(img.width, img.height)
		var s = sizeof(Color)
		curt.cudaMemcpy2D(self.data, self.pitch, img.data, s*img.width, s*img.width, s*img.height, 1)
	end

	-- Copy to a CPU image
	terra CUDAImage:toHostImg(img: &image.Image(real, nchannels))
		img:resize(self.width, self.height)
		var s = sizeof(Color)
		curt.cudaMemcpy2D(img.data, s*img.width, self.data, self.pitch, s*self.width, s*self.height, 2)
	end

	terra CUDAImage:__destruct()
		curt.cudaFree(self.data)
	end

	terra CUDAImage:resize(width: uint, height: uint)
		if self.width ~= width or self.height ~= height then
			self:destruct()
			self:init(width, height)
		end
	end

end)



return CUDAImage