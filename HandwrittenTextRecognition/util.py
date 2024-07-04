import os

base_dir = '/handwriting-recognition'

train_images_dir = os.path.join(base_dir, 'train_v2/train')
validation_images_dir = os.path.join(base_dir, 'validation_v2/validation')
test_images_dir = os.path.join(base_dir, 'test_v2/test')

img_height = img_width = 242

train_length = 9000
validation_length = 2000
test_length = 2000

