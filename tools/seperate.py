import PIL.Image as Image
import numpy as np
import cv2
import os
dirs = os.listdir('images')
print(dirs)
print(len(dirs))
if not os.path.exists('vis'):
    os.makedirs('vis')
if not os.path.exists('ir'):
    os.makedirs('ir')

for img_name in dirs:
    img_pil = Image.open('images/'+img_name)
#img = cv2.imread('images/'+dirs[0], cv2.IMREAD_ANYDEPTH)
    array = np.asarray(img_pil)
    img_rgb = Image.fromarray(array[:, :, :-1])
    img_ir = Image.fromarray(array[:, :, -1])
    img_rgb.save('RGB/'+img_name)
    img_ir.save('IR/'+img_name)

# print(array.shape)
# print(array[:, :, :-1].shape)
# print(array[:, :, -1].shape)