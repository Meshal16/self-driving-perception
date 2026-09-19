import cv2
from torchvision import transforms
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(640, 640))
    
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    print(image.shape, image.min(), image.max())
    
    normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5])
    image = normalize(image)
    return image