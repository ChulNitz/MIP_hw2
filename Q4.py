import idx2numpy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

IMG_BASE_PATH = r"C:\Users\maaya\Documents\Maayan\University\year 4\sem2\medical_image\repo\MIP_hw2"

# device config
device = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# hyper parameters
num_classes = 10 # 0 - 9
num_epochs = 4
batch_size = 100
learning_rate = 0.001
NUM_PHOTOS_TO_TRAIN = 60000
NUM_PHOTOS_TO_TEST = 100


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU())
        self.s2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU())
        self.s4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc = nn.Sequential( 
            nn.Linear(16*5*5, 120),
            nn.ReLU())
        self.fc1 =nn.Sequential( 
            nn.Linear(120, 84),
            nn.ReLU())
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.c1(x)
        out = self.s2(out)
        out = self.c3(out)
        out = self.s4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    

def resize_images(image_arr):
    resized_images = []
    for i in range(image_arr.shape[0]):
        image = np.copy(image_arr[i])
        output_size = (32, 32)
        pad_width = ((output_size[0] - image.shape[0]) // 2,
                    (output_size[1] - image.shape[1]) // 2)
        # pad the image with zeros on all sides
        padded_img_arr = np.zeros(output_size)
        padded_img_arr[pad_width[0]:pad_width[0]+image.shape[0], pad_width[1]:pad_width[1]+image.shape[1]] = image
        resized_images.append(padded_img_arr)
    return resized_images

def train(train_loader):
    model = LeNet5(num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward
            outputs = model(images)
            loss = criterion(outputs,  labels)

            #backward
            optimizer.zero_grad() # clear grad
            loss.backward() # back prop
            optimizer.step() # update weights

            if (i+1) % 100 == 0:
                print(f'epoch  {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():}')
    return model

def test(model, test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        test = 0
        for images, labels in test_loader:
            #images = images.reshape(-1, 28*28).to(device)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # value, index
            _, prediction = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct = (prediction == labels).sum().item()
            #########################Just for fun#############################
            #print("the label is " + str(labels[0]))
            #img = images[0]
            ## convert tensor to numpy array
            #img = img.numpy()
            ## plot image
            #plt.imshow(np.transpose(img, (1, 2, 0)),cmap='gray')
            #plt.show()
            ##################################################################

        acc = 100.0 * n_correct / n_samples
        print(f' accuracy = {acc}')
    return

if __name__ == "__main__":
    # Load training images
    train_images_file = IMG_BASE_PATH + "//train-images.idx3-ubyte"
    train_images_arr = idx2numpy.convert_from_file(train_images_file)
    train_images_arr = train_images_arr[:NUM_PHOTOS_TO_TRAIN]
    resized_train_images = resize_images(train_images_arr)

    # Load training labels
    train_labels_file = IMG_BASE_PATH + "//train-labels.idx1-ubyte"
    train_labels = idx2numpy.convert_from_file(train_labels_file)
    train_labels = train_labels[:NUM_PHOTOS_TO_TRAIN]

    #turn them to dataset and data loader
    data_tensor = torch.from_numpy(np.array(resized_train_images)).float()
    data_tensor = data_tensor.unsqueeze(1)
    labels_tensor = torch.from_numpy(train_labels).long()
    train_dataset = TensorDataset(data_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load testing images
    test_images_file = IMG_BASE_PATH + "//t10k-images.idx3-ubyte"
    test_images_arr = idx2numpy.convert_from_file(test_images_file)
    test_images_arr = test_images_arr[:NUM_PHOTOS_TO_TEST]
    resized_test_images = resize_images(test_images_arr)

    # Load testing labels
    test_labels_file = IMG_BASE_PATH + "//t10k-labels.idx1-ubyte"
    test_labels = idx2numpy.convert_from_file(test_labels_file)
    test_labels=test_labels[:NUM_PHOTOS_TO_TEST]

    #turn them to dataset and data loader
    data_tensor = torch.from_numpy(np.array(resized_test_images)).float()
    data_tensor = data_tensor.unsqueeze(1)
    labels_tensor = torch.from_numpy(test_labels).long()
    test_dataset = TensorDataset(data_tensor, labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = train(train_loader)
    test(model,test_loader)

