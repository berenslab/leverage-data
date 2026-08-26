
from .nako import get_nako_paths, IMAGE_TYPES, NakoDataset, NAKONewDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.helpers import aug2string
import pandas as pd

def load_nako( image_size = 224, batch_size = 512, 
              augment_train = None, 
              label_name = 'basis_uort', 
              contrastive = False, return_loader = True, 
              augment_test = None, augment_val = None,
              normalize = True, return_all = False, 
              return_ids = False,**kwargs):
    image_dir = get_nako_paths(dataset='590', img_res=image_size)



    # base_image_dir = os.path.join(LOWRES_IMAGES_DIR, str(224), 'rt_leftcentral') # Only for nako

    if augment_train is None:
        if normalize:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                        transforms.Normalize(mean=[0.419, 0.209, 0.122],
                                                             std = [0.280, 0.164, 0.113] )
                                        # transforms.Normalize(mean = [0.417, 0.201, 0.114],
                                        #                      std =  [0.265, 0.140, 0.090])
                                         
                                         ])
        else:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                         ])
    transforms_ = f"{aug2string(augment_train)}"  

    if augment_test is None:
        augment_test = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    if augment_val is None:
        augment_val = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])



    dataset_train = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_train, split='train', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)
    dataset_test = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_test, split='test', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)
    dataset_val = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_val, split='val', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)

    train_loader = DataLoader(dataset_train, batch_size = batch_size, num_workers = 20, shuffle= True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=20, shuffle=False)  
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=20, shuffle=False) 
    
    print(f"Dataset  loaded with {len(dataset_train)} training samples, {len(dataset_test)} test samples, and {len(dataset_val)} validation samples.")

    if return_all:
        print('returning all')
        dataset_train_all = NakoDataset(image_dir, IMAGE_TYPES, transform=augment_train, split='all', return_ids=return_ids, feature_name=label_name, contrastive=contrastive)
        train_loader_all = DataLoader(dataset_train_all, batch_size = batch_size, num_workers = 20, shuffle= True)
        if return_loader is not True:
            print('returning all datasets')
            return dataset_train_all, dataset_train, dataset_test, dataset_val, transforms_
        else: 
            print('returning all dataloaders')
            return train_loader_all, train_loader, test_loader, (val_loader, dataset_val), transforms_

    
    if return_loader is not True:
        return dataset_train, dataset_test, dataset_val, transforms_
    print(f"return loader  is {return_loader} therefore returning ify")
    return train_loader, test_loader, (val_loader, dataset_val), transforms_


def load_new_nako( image_size = 224, augment_train = None, normalize = True, ):
    
    root = '/home/berens/bep973/ifeoma_home/data/NAKO/NAKO_macula'
    image_folder = f'{root}/NAKO_macula'
    all_data = f'{root}/nako_reports_macula_and_grade.csv'
    vanilla_train = pd.read_csv(all_data)
    train_ssl = vanilla_train
    val = train_ssl
    test = train_ssl

    if augment_train is None:
        if normalize:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                        transforms.Normalize(mean=[0.419, 0.209, 0.122],
                                                             std = [0.280, 0.164, 0.113] )
                                        # transforms.Normalize(mean = [0.417, 0.201, 0.114],
                                        #                      std =  [0.265, 0.140, 0.090])
                                         
                                         ])
        else:
            augment_train = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(image_size), 
                                        #  transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),  
                                        transforms.RandomHorizontalFlip(),
                                        #  transforms.RandomGrayscale(p=0.2),
                                         ])
    
    dataset_train = NAKONewDataset(image_folder, train_ssl, transform = augment_train)
    return dataset_train


