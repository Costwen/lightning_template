from datasets import load_dataset
import torchvision.transforms as T
from einops import rearrange
import torch.utils.data as data
import requests
from io import BytesIO  
import numpy as np
from PIL import Image
import logging
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.CRITICAL)

class LaionDataset(data.Dataset):
    def __init__(self, name, image_size, mode="train", transform="default"):
        super().__init__()
        self.dataset = load_dataset(name, cache_dir="data")[mode]
        #self.dataset = load_dataset(name)[mode]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5]),
            T.Lambda(lambda x: 2*x-1),
        ])
        if transform == "unique":
            self.transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            url = self.dataset[idx]["URL"]
            text = self.dataset[idx]["TEXT"]
            # read image
            response = requests.get(url, timeout=4)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            # save image
            image = self.transform(image)
            if text is None:
                text = ""
            if image is None:
                return self.__getitem__(np.random.randint(0, len(self)))
        except:
            return self.__getitem__(np.random.randint(0, len(self)))
        return {"image": image, "text": text}

if __name__ == "__main__":
    dataset = LaionDataset("ChristophSchuhmann/improved_aesthetics_5plus", 256)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        image = batch["image"]
        text = batch["text"]
        print(image.shape)
        print(text)
        break  
