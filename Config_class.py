class Config(object):
	def __init__(self,path):
		mod = __import__(path)
		self.blur_kernel_size = mod.blur_kernel_size
		self.threshold_type = mod.threshold_type
		self.threshold_argument = mod.threshold_argument
		self.mode = mod.mode
		self.method = mod.method
		self.erode_kernel_size = mod.erode_kernel_size
		self.number_iterations = mod.number_iterations
		self.x_ratio = mod.x_ratio
		self.y_ratio = mod.y_ratio