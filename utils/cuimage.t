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
		height: uint
	}

	terra CUDAImage:__init() : {}
		self.width = 0
		self.height = 0
		self.data = nil
	end	

	terra CUDAImage:__init(width: uint, height: uint) : {}
		self:__init()
		self.width = width
		self.height = height
		if width*height > 0 then
			curt.cudaMalloc([&&opaque](&self.data), sizeof(Color)*width*height)
		end
	end

	-- Copy from a CPU image
	terra CUDAImage:__init(img: &image.Image(real, nchannels)) : {}
		self:__init(img.width, img.height)
		curt.cudaMemcpy(self.data, img.data, sizeof(Color)*self.width*self.height, 1)
	end

	-- Copy to a CPU image
	terra CUDAImage:toHostImg(img: &image.Image(real, nchannels))
		img:resize(self.width, self.height)
		curt.cudaMemcpy(img.data, self.data, sizeof(Color)*self.width*self.height, 2)
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