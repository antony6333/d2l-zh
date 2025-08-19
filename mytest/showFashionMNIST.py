from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

root = r"d:\dev\test\d2l-zh\data"
ds = datasets.FashionMNIST(root=root, train=True, download=False, transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=10, shuffle=False)

text_labels = ['t-shirt','trouser','pullover','dress','coat',
               'sandal','shirt','sneaker','bag','ankle boot']
imgs, labels = next(iter(loader))  # 取前一批（10張）
for i in range(10):
    img = imgs[i].squeeze().numpy()  # 28x28
    plt.subplot(2,5,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(text_labels[labels[i].item()])
    plt.axis('off')
plt.show()