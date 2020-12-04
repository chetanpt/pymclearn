import sys, os
from mnist import load_mnist
(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()
    with open(save_file, 'rb') as f: 
        dataset = pickle.load(f)

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")



sys.path.append(os.pardir) 
import numpy as np
from mnist import load_mnist
from PIL import Image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape) 
img = img.reshape(28, 28) 
print(img.shape)  

img_show(img)
