# 刘氏生
# 时间：2022/8/3 22:18
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter("../logs")
image_path= "../dataset/val/ants/11381045_b352a47d8c.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("train",img_array,1,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
writer.close()
