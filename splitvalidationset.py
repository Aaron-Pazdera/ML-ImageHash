import glob
import os


from_folder = 'TrainDataset'
to_folder = 'ValidDataset'

file_list = glob.glob(f'{from_folder}/*.png')


for i in range(0, len(file_list), 4):
	file = file_list[i]
	img = file.split(os.sep)[1]

	img_from = f'{from_folder}{os.sep}{img}'
	img_to   = f'{to_folder}{os.sep}{img}'
	print(f'Moving {img_from} to {img_to}')
	os.rename(img_from, img_to)