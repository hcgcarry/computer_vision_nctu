import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

from models import basemodel
from cub2010 import Cub2010

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(dataset_path):
    img_final_height = int(375*0.8)
    img_final_width = int(500*0.8)
    transform_test= transforms.Compose([
        transforms.Resize((img_final_height,img_final_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
            
    testset = Cub2010(root=dataset_path, is_train=False, transform=transform_test)

    BATCH_SIZE = 16
    NUM_WORKERS = 0

    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return test_loader, testset

def test_result(model, test_loader, testset,dataset_path):
    resultDict={}
    with torch.no_grad():
        for i, (inputs, imageNames) in enumerate(test_loader, 0):

            inputs = inputs.to(device)

            outputs,_ = model(inputs)
            _, pred = outputs.max(1)

            for idx in range(len(inputs)):
                resultDict[imageNames[idx]] = int(pred[idx].cpu().numpy())
    
    with open(os.path.join(dataset_path,"testing_img_order.txt")) as f:
        test_images = [x.strip() for x in f.readlines()]  # all the testing images

    submission = []
    for img in test_images:  # image order is important to your result
        predicted_class = resultDict[img]  # the predicted category
        submission.append([img, testset.labelValue2Label(predicted_class)])

    np.savetxt('answer.txt', submission, fmt='%s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch inference')
    parser.add_argument('--dataset', 
                        help="path to dataset")
    # Load the parameters from json file
    args = parser.parse_args()


    test_loader, testset = load_data(args.dataset)

    model_dir = './experiments/baseline/resnet50/weights/weights.pt'
    model = basemodel.resnet50(num_classes=200)
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval() 

    resultDict = test_result(model, test_loader, testset,args.dataset)