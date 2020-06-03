from Detect_space_module import * 
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',default='single')
	parser.add_argument('--image_path',default= '')
	parser.add_argument('--folder_path',default='')
	parser.add_argument('--config_path',default='')
	args = parser.parse_args()
	cfg = Config(args.config_path)
	if args.mode not in ['single','multiple']:
		sys.exit()
	elif args.mode == 'single':
		Process_in_single_cfg(args.image_path,cfg,True)
	else:
		for name in listdir(args.folder_path):
			path = osp.join(args.folder_path,name)
			Process_in_single_cfg(path,cfg,True)