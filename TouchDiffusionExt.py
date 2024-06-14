from TDStoreTools import StorageManager
import TDFunctions as TDF
import numpy as np
import torch
import os 
import webbrowser
import json
from datetime import datetime
import webbrowser

try:
	from StreamDiffusion.utils.wrapper import StreamDiffusionWrapper
except Exception as e:
	current_time = datetime.now()
	formated_time = current_time.strftime("%H:%M:%S")
	op('fifo1').appendRow([formated_time, 'Error', e])


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
		self.rgba_tensor = torch.zeros((512, 512, 4), dtype=torch.float32).to(self.device) #512,768
		self.rgba_tensor[..., 3] = 0
		self.output_interface = TopCUDAInterface(512,512,4,np.float32) #768,512
		self.stream = None

	def activate_stream(self):
		self.update_size()

		acceleration_lora = op('parameter1')['Accelerationlora',1].val
		if acceleration_lora == 'LCM':
			use_lcm_lora = True
		else:
			use_lcm_lora = False
		
		try:
			self.stream = StreamDiffusionWrapper(
				model_id_or_path=f"{op('parameter1')['Checkpoint',1].val}",
				lora_dict=op('parameter1')['Loralist',1].val,
				t_index_list=self.generate_t_index_list(),
				frame_buffer_size=1,
				width= int(op('parameter1')['Sizex',1]),
				height=int(op('parameter1')['Sizey',1]),
				warmup=0,
				acceleration="tensorrt",
				mode= op('parameter1')['Checkpointmode',1].val,
				use_denoising_batch=True,
				cfg_type="self",
				seed=int(op('parameter1')['Seed',1]),
				use_lcm_lora=use_lcm_lora,
				output_type='pt',
				model_type=op('parameter1')['Checkpointtype',1].val,
				touchdiffusion=True,
				#turbo=False
			)

			self.stream.prepare(
				prompt = parent().par.Prompt.val,
				negative_prompt = parent().par.Negprompt.val,
				guidance_scale=parent().par.Cfgscale.val,
				delta=parent().par.Deltamult.val,
				t_index_list=self.update_denoising_strength()
			)

			self.fifolog('Status', 'Engine activated')
		except Exception as e:
			self.fifolog('Error', e)
	
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
	
	def update_size(self):
		width = int(op('parameter1')['Sizex',1])
		height = int(op('parameter1')['Sizey',1])
		print(width,height)
		self.rgba_tensor = torch.zeros((height, width, 4), dtype=torch.float32).to(self.device)
		self.rgba_tensor[..., 3] = 0
		self.output_interface = TopCUDAInterface(width,height,4,np.float32)

	
	def preprocess_image(self, image):
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
		acceleration_mode = parent().par.Acceleration.val 
		if acceleration_mode == 'LCM':
			lcm = True
		if acceleration_mode == 'sd_turbo':
			turbo = True

		return lcm, turbo


	def update_engines(self):
		menuNames = []
		menuLabels = []
		for root, dirs, files in os.walk('engines'):
			if 'unet.engine' in files:
				folder_name = os.path.basename(root)
				split_folder_name = folder_name.split('--')
				if len(split_folder_name) >= 10:
					name = [split_folder_name[0], 
						split_folder_name[2], 
						split_folder_name[3],
						split_folder_name[5]]
					name = '-'.join(name)
					menuLabels.append(name)
					menuNames.append(folder_name)
		
		parent().par.Enginelist.menuNames = menuNames
		parent().par.Enginelist.menuLabels = menuLabels
		self.update_selected_engine()
	
	def update_selected_engine(self):
		vals = parent().par.Enginelist.val.split('--')
		parent().par.Checkpoint = vals[0]
		parent().par.Checkpointtype = vals[1]
		parent().par.Accelerationlora = vals[4]
		parent().par.Checkpointmode = vals[7]
		parent().par.Controlnet = vals[9]
		parent().par.Loralist = vals[8]
		parent().par.Sizex = vals[2]
		parent().par.Sizey = vals[3]
		parent().par.Batchsizex = vals[5]
		parent().par.Batchsizey = vals[6]



	def update_prompt(self):
		prompt = parent().par.Prompt.val
		self.stream.touchdiffusion_prompt(prompt)
	
	def prompt_to_str(self):
		prompt_list = []
		seq = parent().seq.Promptblock
		enable_weights = parent().par.Enableweight
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
		seq = parent().seq.Schedulerblock
		for block in seq.blocks:
			t_index_list.append(block.par.Step)
		self.stream.touchdiffusion_scheduler(t_index_list)
	
	def update_denoising_strength(self):
		amount = parent().par.Denoise
		mode = parent().par.Denoisemode
		#self.stream.touchdiffusion_generate_t_index_list(amount, mode)
		t_index_list = self.stream.touchdiffusion_generate_t_index_list(amount, mode)
		return t_index_list

	def generate_t_index_list(self):
		batchsize = op('parameter1')['Batchsizex',1]
		t_index_list = []
		for i in range(int(batchsize)):
			t_index_list.append(i)
		return t_index_list


	def update_cfg_setting(self):
		guidance_scale = parent().par.Cfgscale
		delta = parent().par.Deltamult.val
		self.stream.touchdiffusion_update_cfg_setting(guidance_scale=guidance_scale, delta=delta)

	def update_noise(self):
		seed = parent().par.Seed.val
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
			self.activate_stream()
		elif par.name == 'Refreshenginelist':
			self.update_engines()
		if par.name[0:3] == 'Url':
			self.about(par.name)
	
	def fifolog(self, status, message):
		current_time = datetime.now()
		formated_time = current_time.strftime("%H:%M:%S")
		op('fifo1').appendRow([formated_time, status, message])
	
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