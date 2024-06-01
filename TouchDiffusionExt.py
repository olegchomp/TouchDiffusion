from TDStoreTools import StorageManager
import TDFunctions as TDF
from utils.wrapper import StreamDiffusionWrapper
import numpy as np
import torch
import os 
import webbrowser
import json

class TouchDiffusionExt:
	"""
	DefaultExt description
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp

		self.source = op('null1')
		self.device = "cuda"
		self.to_tensor = TopArrayInterface(self.source)
		self.stream_toparray = torch.cuda.current_stream(device=self.device)
		self.rgba_tensor = torch.zeros((512, 512, 4), dtype=torch.float32).to(self.device)
		self.rgba_tensor[..., 3] = 1
		self.output_interface = TopCUDAInterface(512,512,4,np.float32)
		self.stream = None

	def activate_stream(self, turbo, lcm):
		_t_index_list = self.generate_t_index_list()
		self.stream = StreamDiffusionWrapper(
			model_id_or_path=self.ownerComp.par.Enginelist.val,
			lora_dict=None,
			t_index_list=_t_index_list,
			frame_buffer_size=1,
			width=512,
			height=512,
			warmup=0,
			acceleration="tensorrt",
			mode="img2img",
			use_denoising_batch=True,
			cfg_type="none",
			seed=self.ownerComp.par.Seed.val,
			use_lcm_lora=lcm,
			output_type='pt',
			touchdiffusion=True,
			turbo=turbo
		)

		self.stream.prepare(
			prompt = self.ownerComp.par.Prompt.val,
			negative_prompt = self.ownerComp.par.Negprompt.val,
			guidance_scale=self.ownerComp.par.Cfgscale.val,
			delta=self.ownerComp.par.Deltamult.val,
			t_index_list=self.update_denoising_strength()
		)
	
	def generate(self, scriptOp):
		stream = self.stream
		self.to_tensor.update(self.stream_toparray.cuda_stream)
		image = torch.as_tensor(self.to_tensor, device=self.device)
		image_tensor = self.preprocess_image(image)

		if hasattr(self.stream, 'batch_size'):
			last_element = 1 if stream.batch_size != 1 else 0
			for _ in range(stream.batch_size - last_element):
				output_image = stream(image=image_tensor)
			
			output_tensor = self.postprocess_image(output_image)
			scriptOp.copyCUDAMemory(
				output_tensor.data_ptr(), 
				self.output_interface.size,  
				self.output_interface.mem_shape)
	
	def preprocess_image(self,image):
		image = torch.flip(image, [1])
		image = torch.clamp(image, 0, 1)
		image = image[:3, :, :] 
		_, h, w = image.shape
		# Resize to integer multiple of 32
		h, w = map(lambda x: x - x % 32, (h, w))
		#image = self.blend_tensors(self.prev_frame, image, 0.5)
		image = image.unsqueeze(0)
		return image

	def postprocess_image(self, image):
		image = torch.flip(image, [1])
		image = image.permute(1, 2, 0)
		self.rgba_tensor[..., :3] = image
		return self.rgba_tensor
	
	def acceleration_mode(self):
		turbo = False
		lcm = False
		acceleration_mode = self.ownerComp.par.Acceleration.val 
		if acceleration_mode == 'LCM':
			lcm = True
		if acceleration_mode == 'sd_turbo':
			turbo = True

		return lcm, turbo


	def update_engines(self):
		op('table1').clear()
		engines_dir = f'{self.ownerComp.par.Venvpath.val}/engines'
		engine_folders = [os.path.join(engines_dir, folder) for folder in os.listdir(engines_dir) if os.path.isdir(os.path.join(engines_dir, folder))]
		if len(engine_folders) > 0:
			parsed_folders = []
			for engine_folder in engine_folders:
				children_folders = [os.path.join(engine_folder, child) for child in os.listdir(engine_folder) if os.path.isdir(os.path.join(engine_folder, child))]
				for child_folder in children_folders:
					parsed_folders.append(self.parse_folder(os.path.relpath(child_folder, engines_dir)))

		else:
			op('table1').appendRow([None, None, None])
	
	def parse_folder(self, folder_path):
		components = folder_path.split('--')
		engine_name = components[0].replace('\\', '/')
		lcm_lora = components[1].split('-')[1]
		max_batch = components[4].split('-')[1]

		# Create dictionary
		folder_dict = {
			'engine_name': engine_name,
			'lcm_lora': lcm_lora,
			'max_batch': max_batch,
			'engine_path': folder_path
		}
		op('table1').appendRow([engine_name, lcm_lora, max_batch, folder_path])
		return folder_dict

	def update_prompt(self):
		prompt = self.ownerComp.par.Prompt.val
		self.stream.touchdiffusion_prompt(prompt)
	
	def prompt_to_str(self):
		prompt_list = []
		seq = self.ownerComp.seq.Promptblock
		enable_weights = self.ownerComp.par.Enableweight
		for block in seq.blocks:
			if block.par.Weight.val > 0:
				if enable_weights:
					prompt_with_weight = f'({block.par.Prompt.val}){block.par.Weight.val}'
				else:
					prompt_with_weight = block.par.Prompt.val

				prompt_list.append(prompt_with_weight)
		
		prompt_str = ", ".join(prompt_list)
		return prompt_str

	def update_scheduler(self):
		t_index_list = []
		seq = self.ownerComp.seq.Schedulerblock
		for block in seq.blocks:
			t_index_list.append(block.par.Step)
		self.stream.touchdiffusion_scheduler(t_index_list)
	
	def update_denoising_strength(self):
		amount = self.ownerComp.par.Denoise
		mode = self.ownerComp.par.Denoisemode
		#self.stream.touchdiffusion_generate_t_index_list(amount, mode)
		t_index_list = self.stream.touchdiffusion_generate_t_index_list(amount, mode)
		return t_index_list

	def generate_t_index_list(self):
		t_index_list = []
		for i in range(int(self.ownerComp.par.Batchsize.val)):
			t_index_list.append(i)
		return t_index_list


	def update_cfg_setting(self):
		guidance_scale = self.ownerComp.par.Cfgscale
		delta = self.ownerComp.par.Deltamult.val
		self.stream.touchdiffusion_update_cfg_setting(guidance_scale=guidance_scale, delta=delta)

	def update_noise(self):
		seed = self.ownerComp.par.Seed.val
		self.stream.touchdiffusion_update_noise(seed=seed)

	
	def parexec_onValueChange(self, par, prev):
		if hasattr(self.stream, 'batch_size'):
			if par.name == 'Prompt':
				self.update_prompt()
			elif par.name == 'Denoise':
				self.update_denoising_strength()
			elif par.name == 'Cfgscale':
				self.update_cfg_setting()
			elif par.name == 'Seed':
				self.update_noise()
	
	def parexec_onPulse(self, par):
		if par.name == 'Loadengine':
			lcm, turbo = self.acceleration_mode()
			self.activate_stream(turbo, lcm)
		elif par.name == 'Refreshenginelist':
			self.update_engines()
		if par.name[0:3] == 'Url':
			self.about(par.name)
	
	def about(self, endpoint):
		if endpoint == 'Urlg':
			webbrowser.open('https://github.com/olegchomp/TouchDiffusion', new=2)
		if endpoint == 'Urld':
			webbrowser.open('https://discord.gg/wNW8xkEjrf', new=2)
		if endpoint == 'Urlt':
			webbrowser.open('https://www.youtube.com/vjschool', new=2)
		if endpoint == 'Urla':
			webbrowser.open('https://olegcho.mp/', new=2)
		if endpoint == 'Urldonate':
			webbrowser.open('https://boosty.to/vjschool/donate', new=2)

	def version(self):
		url = 'https://api.github.com/repos/olegchomp/TDComfyUI/releases/latest'
		try:
			def callback(statusCode, headerDict, data, id):

				if statusCode['code'] == 200:
					gitversion = json.loads(data)['name'].split('v.')[1]
					if gitversion == self.tdcomfy.par.Version:
						self.history("Status: Component version up to date")
					else:
						self.history("Status: New version available on Github!")
				else:
					self.history("Status: Component version not checked")

			op.TDResources.WebClient.Request(callback, url, "GET")
		except:
			self.history("Status: Component version not checked")

class TopCUDAInterface:
	def __init__(self, width, height, num_comps, dtype):
		self.mem_shape = CUDAMemoryShape()
		self.mem_shape.width = width
		self.mem_shape.height = height
		self.mem_shape.numComps = num_comps
		self.mem_shape.dataType = dtype
		self.bytes_per_comp = np.dtype(dtype).itemsize
		self.size = width * height * num_comps * self.bytes_per_comp

class TopArrayInterface:
	def __init__(self, top, stream=0):
		self.top = top
		mem = top.cudaMemory(stream=stream)
		self.w, self.h = mem.shape.width, mem.shape.height
		self.num_comps = mem.shape.numComps
		self.dtype = mem.shape.dataType
		shape = (mem.shape.numComps, self.h, self.w)
		dtype_info = {'descr': [('', '<f4')], 'num_bytes': 4}
		dtype_descr = dtype_info['descr']
		num_bytes = dtype_info['num_bytes']
		num_bytes_px = num_bytes * mem.shape.numComps
		
		self.__cuda_array_interface__ = {
			"version": 3,
			"shape": shape,
			"typestr": dtype_descr[0][1],
			"descr": dtype_descr,
			"stream": stream,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	def update(self, stream=0):
		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__['stream'] = stream
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
		return
