import importlib.util

class Config(object):
	def __init__(self,path):
		spec = importlib.util.spec_from_file_location("module.name", path)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		self.blur_kernel_size = mod.blur_kernel_size
		self.threshold_type = mod.threshold_type
		self.threshold_argument = mod.threshold_argument
		self.mode = mod.mode
		self.method = mod.method
		self.erode_kernel_size = mod.erode_kernel_size
		self.number_iterations = mod.number_iterations
		self.x_ratio = mod.x_ratio
		self.y_ratio = mod.y_ratio
		self.contour_limit = mod.contour_limit