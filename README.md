# 3D Cloud Classification : PointNet apply to ModelNet10


```python
#import all the necessary stuffs
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  
global device
print(device)
```

    cuda:0



```python
from torch.utils.tensorboard import SummaryWriter

#
#
#      0===============================0
#      |    PLY files reader/writer    |
#      0===============================0
#
#
#------------------------------------------------------------------------------------------
#
#      function to read/write .ply files
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 10/02/2017
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Basic libs
import numpy as np
import sys


# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#

def parse_header(plyfile, ext):

    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def read_ply(filename):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:
        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # Parse header
        num_points, properties = parse_header(plyfile, ext)

        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)
    return data


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will
        be appended to the file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of
        numpy arrays or a tuple of numpy arrays. Each 1D numpy array and each
        column of 2D numpy arrays are considered as one field.

    field_names : list
        the name of each fields as a list of strings. Has to be the same length
        as the number of fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field is None:
            print('WRITE_PLY ERROR: a field is None')
            return False
        elif field.ndim > 2:
            print('WRITE_PLY ERROR: a field have more than 2 dimensions')
            return False
        elif field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element
```


```python
#define default and custom transformation for 3D cloud objects

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud).to(torch.float32).to(device)
    
class RandomRotation_z(object):
    def __call__(self, pointcloud):
        
        theta = random.random() * 2. * math.pi
        rot_matrix = torch.tensor([[math.cos(theta), -math.sin(theta),      0],
                               [math.sin(theta),  math.cos(theta),      0],
                               [0,                              0,      1]],dtype=torch.float32).to(device)
        rot_pointcloud = torch.matmul(pointcloud,rot_matrix)
        return rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        
        noise = torch.rand(pointcloud.size(0),pointcloud.size(1)).to(device)*0.02
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
    
class ShufflePoints(object):
    def __call__(self, pointcloud):
        index = torch.randperm(pointcloud.size(0))
        pointcloud[:] = pointcloud[index]
        return pointcloud
    
class AxisReducer(object):
    def __call__(self, pointcloud):
        pointcloud[:,0] = torch.sqrt(torch.square(pointcloud[:,0]) + torch.square(pointcloud[:,1]))
        pointcloud[:,1] = 0
        return pointcloud
    
class NormalizePoints(object):
    def __call__(self, pointcloud):
        return pointcloud/(torch.max(torch.min(pointcloud),torch.max(pointcloud)))
    
class PointsToVoxel(object):
    def __call__(self, pointcloud,voxel_size = 8):
        pointcloud= (((pointcloud+1)/2.01)*voxel_size).int()
        return pointcloud

class VoxelToBool(object): #to visualize the result of PointsToVoxel
    def __call__(self, pointcloud,voxel_size = 8):
        bool_array = torch.zeros((voxel_size,voxel_size,voxel_size,),dtype=bool)
        for i in range(pointcloud.size(0)):
            bool_array[pointcloud[i][0],pointcloud[i][1],pointcloud[i][2]]=True
        return bool_array

def default_transforms():
    return transforms.Compose([
        ToTensor(),
        RandomRotation_z(),
        RandomNoise(),
        ShufflePoints(),
        ])
                              
def customize_transforms():
    return transforms.Compose([
        ToTensor(),
        RandomRotation_z(),
        RandomNoise(),
        AxisReducer(),
        ShufflePoints(),
        ])

def customize_transforms_voxel():
    return transforms.Compose([
        ToTensor(),
        RandomRotation_z(),
        RandomNoise(),
        NormalizePoints(),
        PointsToVoxel(),
        ShufflePoints(),
        ])

```


```python
#define our and verify our dataset
class PointCloudData(Dataset):
    def __init__(self,
                 root_dir,
                 folder="train",
                 transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir))
                   if os.path.isdir(root_dir + "/" + dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    sample = {}
                    sample['ply_path'] = new_dir+"/"+file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]['ply_path']
        category = self.files[idx]['category']
        data = read_ply(ply_path)
        pointcloud = self.transforms(np.vstack((data['x'],
                                                data['y'],
                                                data['z'])).T)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}
    

def slice_dataset(dataset_input):
    dataset = []
    for i in range(10):
        dataset.append([])

    for obj in dataset_input:
        index = obj["category"]
        dataset[index].append(obj["pointcloud"])
    torch_dataset = []
    for data in dataset:
        #print(torch.cat(data))
       
        torch_dataset.append(torch.stack(data))
        
    return torch_dataset



NUM_POINTS = 1024
NUM_CLASSES = 10
BATCH_SIZE = 32


train_ds = PointCloudData("./data/ModelNet10_PLY")
test_ds = PointCloudData("./data/ModelNet10_PLY", folder='test')


dataloader_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)



train_ds_augment = PointCloudData("./data/ModelNet10_PLY",transform=customize_transforms())
test_ds_augment = PointCloudData("./data/ModelNet10_PLY", folder='test',transform=customize_transforms())

dataloader_train_augment = DataLoader(train_ds_augment, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test_augment = DataLoader(test_ds_augment, batch_size=BATCH_SIZE, shuffle=True)

train_ds_augment_voxel = PointCloudData("./data/ModelNet10_PLY",transform=customize_transforms_voxel())
test_ds_augment_voxel = PointCloudData("./data/ModelNet10_PLY", folder='test',transform=customize_transforms_voxel())

dataloader_train_augment_voxel = DataLoader(train_ds_augment_voxel, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test_augment_voxel = DataLoader(test_ds_augment_voxel, batch_size=BATCH_SIZE, shuffle=True)




index =800


plt.figure(1)
points = train_ds[index]["pointcloud"].cpu()


plt.figure(1)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2],s=5)
ax.set_axis_off()
plt.show()



points = train_ds_augment_voxel[index]["pointcloud"].cpu()

voxel_to_bool = VoxelToBool()

voxelarray = voxel_to_bool(points)
print(voxelarray.shape)


plt.rcParams["figure.figsize"] = [5,5]
plt.rcParams["figure.autolayout"] = True
ax = plt.figure(2).add_subplot(projection='3d')



#ax.voxels(voxelarray, edgecolor="k",facecolors="red")
ax.voxels(voxelarray, edgecolor='k')

plt.show()
print(voxelarray.shape)
```


    <Figure size 640x480 with 0 Axes>



    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_4_1.png)
    


    torch.Size([8, 8, 8])


    /home/a/.local/lib/python3.8/site-packages/IPython/core/pylabtools.py:152: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_4_4.png)
    


    torch.Size([8, 8, 8])



```python
# show an example of each object
num_classes = 10
titles_object =[]
for key in train_ds_augment.classes:
    titles_object.append(key)

# fig, ax = plt.subplots(num_classes, 5, figsize=(10,20))
fig = plt.figure(figsize=(14,20))

sliced_dataset = slice_dataset(train_ds)
sliced_dataset_test = slice_dataset(test_ds)

size_x = 7
for i in range(num_classes):
    for j in range(size_x):
        points = sliced_dataset[i][j].cpu()
        
        ax = fig.add_subplot(num_classes, size_x, 1+i*size_x+j, projection='3d')
        plt.title(titles_object[i]+" : "+str(j))

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        
        
        ax.set_axis_off()

plt.legend()
plt.show()

print("Numbers of each object")
for i in range(num_classes):
    print(i,len(sliced_dataset[i]),len(sliced_dataset_test[i]))
    
del sliced_dataset,sliced_dataset_test

```

    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_5_1.png)
    


    Numbers of each object
    0 106 50
    1 515 100
    2 889 100
    3 200 86
    4 200 86
    5 465 100
    6 200 86
    7 680 100
    8 392 100
    9 344 100



```python
#define plot, loss and train loop of our dataset
def plot_all(accuracy_train_array,accuracy_test_array,loss_train_array,loss_test_array):
    plt.figure(1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy in %")
    plt.plot(accuracy_train_array,label="Accuracy train")
    plt.plot(accuracy_test_array,label="Accuracy test")

    plt.legend()
    plt.show()
    plt.figure(2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(loss_train_array,label="Loss train")
    plt.plot(loss_test_array,label="Loss test")

    plt.legend()
    plt.show()
def basic_loss(outputs, labels):
    #outputs,labels =  torch.Tensor(outputs,dtype=torch.long),torch.Tensor(labels,dtype=torch.long)
    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.CrossEntropyLoss()
    bsize = outputs.size(0)
    #outputs = torch.transpose(outputs,0, 1)
    return criterion(outputs, labels)


def pointnet_full_loss(outputs, labels, m1, m2, loss_func, alpha=0.001):
    #criterion = torch.nn.NLLLoss()
    criterion = loss_func
    #criterion = torch.nn.CrossEntropyLoss()
    bsize = outputs.size(0)

    
    id_1 = torch.eye(m1.size(1), requires_grad=True).repeat(bsize, 1, 1).to(device)
    diff1 = id_1 - torch.bmm(m1, m1.transpose(1, 2))
    
    id_2 = torch.eye(m2.size(1), requires_grad=True).repeat(bsize, 1, 1).to(device)
    diff2 = id_2 - torch.bmm(m2, m2.transpose(1, 2))
    
    
    return criterion(outputs, labels) + alpha * (torch.norm(diff1)) / float(bsize) + alpha * (torch.norm(diff2)) / float(bsize)

def get_accuracy(labels_predict,labels_true):
    _, predicted = torch.max(labels_predict.data, 1)
    total = labels_true.size(0)
    correct = (predicted == labels_true).sum().item()
    val_acc = 100. * correct / total
    return val_acc

def evaluation_model(model,dataloader_test,loss_func):
    correct = total = 0
    loss_test=0
    val_acc_test=0
    size=0
    with torch.no_grad():
        for id_batch, data in enumerate(dataloader_test):
            inputs, labels = data['pointcloud'].float(), data['category']
            size+=1
            predicted,rotation_1,rotation_2 = model(inputs)
                        # outputs, __ = model(inputs.transpose(1,2))

            predicted, labels = torch.Tensor(predicted).type(torch.FloatTensor),torch.Tensor(labels).type(torch.LongTensor)
            #loss_test = basic_loss(predicted, labels)
            loss_test += pointnet_full_loss(predicted, labels,rotation_1,rotation_2,loss_func)
            val_acc_test += get_accuracy(predicted,labels)
        
    return loss_test/size,val_acc_test/size
    

def train(
    model,
    dataloader_train, 
    dataloader_test,  
    epochs=100, 
    loss_func=torch.nn.NLLLoss()
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20, gamma=0.5)
    
    (loss_train_array,accuracy_train_array,loss_test_array,accuracy_test_array,)= [],[],[],[]

    for epoch in range(epochs):
        
        for id_batch, data in enumerate(dataloader_train):
            inputs, labels = data['pointcloud'].float(), data['category']


            optimizer.zero_grad()
            labels_predict,rotation_1,rotation_2 = model(inputs)
            labels_predict, labels = torch.Tensor(labels_predict).type(torch.FloatTensor),torch.Tensor(labels).type(torch.LongTensor)
            #loss_train = basic_loss(labels_predict, labels)

            loss_train = pointnet_full_loss(labels_predict, labels,rotation_1,rotation_2,loss_func)
        
            loss_train.backward()
            optimizer.step()
            #print(id_batch)
            if id_batch==0:

                acc_train = get_accuracy(labels_predict,labels)
                loss_train_array.append(loss_train.cpu().detach().numpy())
                accuracy_train_array.append(acc_train)
                scheduler.step()
                loss_test,acc_test=evaluation_model(model,dataloader_test,loss_func)
                loss_test_array.append(loss_test.cpu().detach().numpy())
                accuracy_test_array.append(acc_test)
                
                print('Epoch: %d, Loss_train: %.3f, Accuracy_train: %.1f %%, Loss_test %.3f, Accuracy_test: %.1f %%' % (epoch+1, loss_train, acc_train,loss_test, acc_test))

    return loss_train_array,accuracy_train_array,loss_test_array,accuracy_test_array


```


```python
#define our different model

class PointMLP(nn.Module):
    def __init__(self, input_size, classes=10):
        super(PointMLP, self).__init__()
        l_1 = 512
        l_2 = 256

        self.fc_1 = nn.Linear(NUM_POINTS*3,l_1).to(device)
        self.bn_1 = nn.BatchNorm1d(l_1).to(device)
        self.fc_2 = nn.Linear(l_1,l_2).to(device)
        self.dropout_1 = nn.Dropout(0.3).to(device)
        self.bn_2 = nn.BatchNorm1d(l_2).to(device)

        self.fc_3 = nn.Linear(l_2,classes).to(device)
        self.bn_3 = nn.BatchNorm1d(classes).to(device)
        
        self.eye_1 = torch.eye(1, requires_grad=False)
        self.eye_2 = torch.eye(1, requires_grad=False)

    
    def forward(self, x):
        x = x.to(device)

        x = torch.flatten(x,start_dim=1)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = self.dropout_1(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        return x,self.eye_1.repeat(x.size(0), 1, 1).to(device),self.eye_2.repeat(x.size(0), 1, 1).to(device)


class Tnet(nn.Module):
    def __init__(self, input_size, kernel_size):
        super(Tnet, self).__init__()
        #l_1 = 64
        #l_2 = 128
        #l_3 = 1024
        #l_4 = 512
        #l_5 = 256

        l_1 = 32
        l_2 = 64
        l_3 = 256
        l_4 = 128
        l_5 = 64

        self.kn_size = kernel_size

        self.fc_1 = nn.Conv1d(kernel_size,l_1,1).to(device)
        self.bn_1 = nn.BatchNorm1d(l_1).to(device)
        self.fc_2 = nn.Conv1d(l_1,l_2,1).to(device)
        self.bn_2 = nn.BatchNorm1d(l_2).to(device)
        self.fc_3 = nn.Conv1d(l_2,l_3,1).to(device)
        self.bn_3 = nn.BatchNorm1d(l_3).to(device)

        self.mp = nn.MaxPool1d(l_3).to(device)
        self.fc_4 = nn.Linear(input_size,l_4).to(device)
        self.bn_4 = nn.BatchNorm1d(l_4).to(device)
        self.fc_5 = nn.Linear(l_4,l_5).to(device)
        self.bn_5 = nn.BatchNorm1d(l_5).to(device)
        self.fc_6 = nn.Linear(l_5,self.kn_size*self.kn_size).to(device)

    def forward(self, x):
        x = x.to(device)

        #x = torch.flatten(x,start_dim=-2)
        x = x.transpose(2, 1)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        x = self.bn_3(x)
        x = F.relu(x)

        x = self.mp(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc_4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.fc_5(x)
        x = self.bn_5(x)
        x = F.relu(x)
        x = self.fc_6(x)
        x = x.view(-1,self.kn_size,self.kn_size)
        return x

class InputTransform(nn.Module):
    def __init__(self,input_size, kernel_size):
        super(InputTransform, self).__init__()
        self.kn_size = kernel_size
        self.eye_init = torch.eye(kernel_size, requires_grad=False)
        self.t_net = Tnet(input_size, kernel_size).to(device)

    def forward(self, x):
        x = x.to(device)
        kern = self.t_net(x) + self.eye_init.repeat(x.size(0), 1, 1).to(device)
        x = torch.matmul(x,kern)
        return x,kern


class PointNetBasic(nn.Module):
    def __init__(self, input_size,classes=10):
        super(PointNetBasic, self).__init__()
        #l_1 = 64
        #l_2 = 64
        #l_3 = 64
        #l_4 = 128
        #l_5 = 1024
        #l_6 = 512
        #l_7 = 256

        l_1 = 32
        l_2 = 32
        l_3 = 32
        l_4 = 128
        l_5 = 512
        l_6 = 256
        l_7 = 128

        self.fc_1 = nn.Conv1d(3,l_1,1).to(device)
        self.bn_1 = nn.BatchNorm1d(l_1).to(device)
        self.fc_2 = nn.Conv1d(l_1,l_2,1).to(device)
        self.bn_2 = nn.BatchNorm1d(l_2).to(device)
        self.fc_3 = nn.Conv1d(l_2,l_3,1).to(device)
        self.bn_3 = nn.BatchNorm1d(l_3).to(device)
        self.fc_4 = nn.Conv1d(l_3,l_4,1).to(device)
        self.bn_4 = nn.BatchNorm1d(l_4).to(device)
        self.fc_5 = nn.Conv1d(l_4,l_5,1).to(device)
        self.bn_5 = nn.BatchNorm1d(l_5).to(device)

        self.mp = nn.MaxPool1d(l_5).to(device)
        self.fc_6 = nn.Linear(input_size,l_6).to(device)
        self.bn_6 = nn.BatchNorm1d(l_6).to(device)
        self.fc_7 = nn.Linear(l_6,l_7).to(device)
        self.dropout_1 = nn.Dropout(0.3).to(device)
        self.bn_7 = nn.BatchNorm1d(l_7).to(device)
        self.fc_8 = nn.Linear(l_7,classes).to(device)
        self.eye_1 = torch.eye(1, requires_grad=False)
        self.eye_2 = torch.eye(1, requires_grad=False)
    
    def forward(self, x):
        x=x.to(device)
        x = x.transpose(2, 1)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.fc_4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.fc_5(x)
        x = self.bn_5(x)
        x = F.relu(x)

        x = self.mp(x)
        x = torch.flatten(x,start_dim=1)

        x = self.fc_6(x)
        x = self.bn_6(x)
        x = F.relu(x)
        x = self.fc_7(x)
        x = self.bn_7(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.fc_8(x)
        return x,self.eye_1.repeat(x.size(0), 1, 1).to(device),self.eye_2.repeat(x.size(0), 1, 1).to(device)




class PointNetFull(nn.Module):
    def __init__(self, input_size,classes=10):
        super(PointNetFull, self).__init__()
        #l_1 = 64
        #l_2 = 64
        #l_3 = 64
        #l_4 = 128
        #l_5 = 1024
        #l_6 = 512
        #l_7 = 256
        
        l_1 = 32
        l_2 = 32
        l_3 = 32
        l_4 = 128
        l_5 = 512
        l_6 = 256
        l_7 = 128
        
        #torch.nn.Conv1d(3, 64, 1)
        self.input_transform_1 = InputTransform(input_size,3).to(device)
        
        self.fc_1 = nn.Conv1d(3,l_1,1).to(device)
        self.bn_1 = nn.BatchNorm1d(l_1).to(device)
        self.fc_2 = nn.Conv1d(l_1,l_2,1).to(device)
        self.bn_2 = nn.BatchNorm1d(l_2).to(device)
        
        self.fc_3 = nn.Conv1d(l_2,l_3,1).to(device)
        self.bn_3 = nn.BatchNorm1d(l_3).to(device)
        self.fc_4 = nn.Conv1d(l_3,l_4,1).to(device)
        self.bn_4 = nn.BatchNorm1d(l_4).to(device)
        self.fc_5 = nn.Conv1d(l_4,l_5,1).to(device)
        self.bn_5 = nn.BatchNorm1d(l_5).to(device)
        
        self.mp = nn.MaxPool1d(l_5).to(device)
        self.fc_6 = nn.Linear(input_size,l_6).to(device)
        self.bn_6 = nn.BatchNorm1d(l_6).to(device)
        self.fc_7 = nn.Linear(l_6,l_7).to(device)
        self.dropout_1 = nn.Dropout(0.3).to(device)
        self.bn_7 = nn.BatchNorm1d(l_7).to(device)
        self.fc_8 = nn.Linear(l_7,classes).to(device)
        self.eye_1 = torch.eye(1, requires_grad=False)
        self.eye_2 = torch.eye(1, requires_grad=False)

    def forward(self, x):
        x=x.to(device)
        x,rotation_1 = self.input_transform_1(x)
        
        x = x.transpose(2, 1)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        
        x = self.fc_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        
        x = self.fc_4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        
        x = self.fc_5(x)
        x = self.bn_5(x)
        x = F.relu(x)
        
        x = self.mp(x)
        x = torch.flatten(x,start_dim=1)
        
        x = self.fc_6(x)
        x = self.bn_6(x)
        x = F.relu(x)
        
        x = self.fc_7(x)
        x = self.bn_7(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.fc_8(x)
        return x,rotation_1,self.eye_1.repeat(x.size(0), 1, 1).to(device)






```

# Results :


```python
#PointMLP, no augment
model = PointMLP(NUM_POINTS,NUM_CLASSES)
(loss_train_array_POINTMLP,
 accuracy_train_array_POINTMLP,
 loss_test_array_POINTMLP,
 accuracy_test_array_POINTMLP) = train(
    model, 
    dataloader_train,
    dataloader_test, 
    epochs=50,
    loss_func=torch.nn.CrossEntropyLoss()
)

plt.figure(1)
plt.title("PointMLP Acurracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")

plt.plot(accuracy_train_array_POINTMLP,label="train_PointMLP")
plt.plot(accuracy_test_array_POINTMLP,label="test_PointMLP")
plt.grid()
plt.legend()


```

    Epoch: 1, Loss_train: 2.378, Accuracy_train: 9.4 %, Loss_test 2.485, Accuracy_test: 11.2 %
    Epoch: 2, Loss_train: 2.371, Accuracy_train: 15.6 %, Loss_test 2.376, Accuracy_test: 14.2 %
    Epoch: 3, Loss_train: 2.240, Accuracy_train: 28.1 %, Loss_test 2.342, Accuracy_test: 14.7 %
    Epoch: 4, Loss_train: 2.276, Accuracy_train: 15.6 %, Loss_test 2.335, Accuracy_test: 12.9 %
    Epoch: 5, Loss_train: 2.168, Accuracy_train: 21.9 %, Loss_test 2.316, Accuracy_test: 13.9 %
    Epoch: 6, Loss_train: 2.208, Accuracy_train: 21.9 %, Loss_test 2.304, Accuracy_test: 13.8 %
    Epoch: 7, Loss_train: 2.188, Accuracy_train: 25.0 %, Loss_test 2.325, Accuracy_test: 12.0 %
    Epoch: 8, Loss_train: 2.241, Accuracy_train: 18.8 %, Loss_test 2.320, Accuracy_test: 13.4 %
    Epoch: 9, Loss_train: 2.177, Accuracy_train: 25.0 %, Loss_test 2.312, Accuracy_test: 14.5 %
    Epoch: 10, Loss_train: 2.167, Accuracy_train: 15.6 %, Loss_test 2.316, Accuracy_test: 12.9 %
    Epoch: 11, Loss_train: 1.984, Accuracy_train: 25.0 %, Loss_test 2.335, Accuracy_test: 13.7 %
    Epoch: 12, Loss_train: 2.064, Accuracy_train: 37.5 %, Loss_test 2.338, Accuracy_test: 13.6 %
    Epoch: 13, Loss_train: 2.093, Accuracy_train: 28.1 %, Loss_test 2.329, Accuracy_test: 13.5 %
    Epoch: 14, Loss_train: 2.044, Accuracy_train: 25.0 %, Loss_test 2.326, Accuracy_test: 12.7 %
    Epoch: 15, Loss_train: 2.046, Accuracy_train: 25.0 %, Loss_test 2.327, Accuracy_test: 13.7 %
    Epoch: 16, Loss_train: 2.148, Accuracy_train: 21.9 %, Loss_test 2.304, Accuracy_test: 15.3 %
    Epoch: 17, Loss_train: 2.138, Accuracy_train: 15.6 %, Loss_test 2.322, Accuracy_test: 15.1 %
    Epoch: 18, Loss_train: 2.223, Accuracy_train: 15.6 %, Loss_test 2.306, Accuracy_test: 15.9 %
    Epoch: 19, Loss_train: 1.959, Accuracy_train: 43.8 %, Loss_test 2.309, Accuracy_test: 14.7 %
    Epoch: 20, Loss_train: 2.011, Accuracy_train: 34.4 %, Loss_test 2.311, Accuracy_test: 14.9 %
    Epoch: 21, Loss_train: 2.015, Accuracy_train: 31.2 %, Loss_test 2.297, Accuracy_test: 16.7 %
    Epoch: 22, Loss_train: 1.880, Accuracy_train: 37.5 %, Loss_test 2.280, Accuracy_test: 16.6 %
    Epoch: 23, Loss_train: 1.995, Accuracy_train: 37.5 %, Loss_test 2.266, Accuracy_test: 16.3 %
    Epoch: 24, Loss_train: 1.920, Accuracy_train: 31.2 %, Loss_test 2.303, Accuracy_test: 15.4 %
    Epoch: 25, Loss_train: 1.620, Accuracy_train: 50.0 %, Loss_test 2.240, Accuracy_test: 18.9 %
    Epoch: 26, Loss_train: 2.038, Accuracy_train: 31.2 %, Loss_test 2.267, Accuracy_test: 18.3 %
    Epoch: 27, Loss_train: 2.057, Accuracy_train: 34.4 %, Loss_test 2.260, Accuracy_test: 18.0 %
    Epoch: 28, Loss_train: 1.951, Accuracy_train: 40.6 %, Loss_test 2.262, Accuracy_test: 17.2 %
    Epoch: 29, Loss_train: 2.038, Accuracy_train: 31.2 %, Loss_test 2.245, Accuracy_test: 18.8 %
    Epoch: 30, Loss_train: 2.129, Accuracy_train: 12.5 %, Loss_test 2.250, Accuracy_test: 18.0 %
    Epoch: 31, Loss_train: 2.108, Accuracy_train: 21.9 %, Loss_test 2.229, Accuracy_test: 19.6 %
    Epoch: 32, Loss_train: 1.909, Accuracy_train: 31.2 %, Loss_test 2.213, Accuracy_test: 20.0 %
    Epoch: 33, Loss_train: 1.931, Accuracy_train: 37.5 %, Loss_test 2.216, Accuracy_test: 20.8 %
    Epoch: 34, Loss_train: 1.992, Accuracy_train: 31.2 %, Loss_test 2.205, Accuracy_test: 19.3 %
    Epoch: 35, Loss_train: 1.924, Accuracy_train: 34.4 %, Loss_test 2.196, Accuracy_test: 20.5 %
    Epoch: 36, Loss_train: 1.753, Accuracy_train: 43.8 %, Loss_test 2.217, Accuracy_test: 19.2 %
    Epoch: 37, Loss_train: 1.926, Accuracy_train: 28.1 %, Loss_test 2.205, Accuracy_test: 19.6 %
    Epoch: 38, Loss_train: 2.000, Accuracy_train: 28.1 %, Loss_test 2.174, Accuracy_test: 19.0 %
    Epoch: 39, Loss_train: 1.571, Accuracy_train: 50.0 %, Loss_test 2.177, Accuracy_test: 20.3 %
    Epoch: 40, Loss_train: 1.758, Accuracy_train: 28.1 %, Loss_test 2.181, Accuracy_test: 19.8 %
    Epoch: 41, Loss_train: 1.577, Accuracy_train: 40.6 %, Loss_test 2.155, Accuracy_test: 20.7 %
    Epoch: 42, Loss_train: 1.862, Accuracy_train: 28.1 %, Loss_test 2.151, Accuracy_test: 21.0 %
    Epoch: 43, Loss_train: 2.096, Accuracy_train: 12.5 %, Loss_test 2.130, Accuracy_test: 20.8 %
    Epoch: 44, Loss_train: 1.865, Accuracy_train: 40.6 %, Loss_test 2.139, Accuracy_test: 21.2 %
    Epoch: 45, Loss_train: 1.953, Accuracy_train: 37.5 %, Loss_test 2.132, Accuracy_test: 21.9 %
    Epoch: 46, Loss_train: 1.733, Accuracy_train: 37.5 %, Loss_test 2.115, Accuracy_test: 20.7 %
    Epoch: 47, Loss_train: 1.754, Accuracy_train: 34.4 %, Loss_test 2.145, Accuracy_test: 20.4 %
    Epoch: 48, Loss_train: 1.805, Accuracy_train: 28.1 %, Loss_test 2.120, Accuracy_test: 20.9 %
    Epoch: 49, Loss_train: 1.833, Accuracy_train: 34.4 %, Loss_test 2.128, Accuracy_test: 21.9 %
    Epoch: 50, Loss_train: 1.900, Accuracy_train: 37.5 %, Loss_test 2.116, Accuracy_test: 22.1 %





    <matplotlib.legend.Legend at 0x7f0fd238eb20>




    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_9_2.png)
    



```python
#PointBasic, no augment
model = PointNetBasic(NUM_POINTS,NUM_CLASSES)
(loss_train_array_POINTNetBasic,
 accuracy_train_array_POINTNetBasic,
 loss_test_array_POINTNetBasic,
 accuracy_test_array_POINTNetBasic) = train(
    model, 
    dataloader_train,
    dataloader_test, 
    epochs=50,
    loss_func=torch.nn.CrossEntropyLoss()
)



plt.figure(1)
plt.title("PointNetBasic Acurracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")

plt.plot(accuracy_train_array_POINTNetBasic,label="train_PointNetBasic")
plt.plot(accuracy_test_array_POINTNetBasic,label="test_PointNetBasic")
plt.grid()
plt.legend()
```

    Epoch: 1, Loss_train: 2.415, Accuracy_train: 12.5 %, Loss_test 2.226, Accuracy_test: 23.0 %
    Epoch: 2, Loss_train: 0.693, Accuracy_train: 71.9 %, Loss_test 0.717, Accuracy_test: 76.7 %
    Epoch: 3, Loss_train: 0.423, Accuracy_train: 90.6 %, Loss_test 0.675, Accuracy_test: 77.2 %
    Epoch: 4, Loss_train: 0.496, Accuracy_train: 84.4 %, Loss_test 0.629, Accuracy_test: 79.6 %
    Epoch: 5, Loss_train: 0.426, Accuracy_train: 84.4 %, Loss_test 0.557, Accuracy_test: 82.0 %
    Epoch: 6, Loss_train: 0.210, Accuracy_train: 96.9 %, Loss_test 0.565, Accuracy_test: 81.9 %
    Epoch: 7, Loss_train: 0.215, Accuracy_train: 90.6 %, Loss_test 0.517, Accuracy_test: 83.5 %
    Epoch: 8, Loss_train: 0.166, Accuracy_train: 96.9 %, Loss_test 0.532, Accuracy_test: 83.4 %
    Epoch: 9, Loss_train: 0.124, Accuracy_train: 100.0 %, Loss_test 0.571, Accuracy_test: 79.3 %
    Epoch: 10, Loss_train: 0.260, Accuracy_train: 87.5 %, Loss_test 0.476, Accuracy_test: 85.0 %
    Epoch: 11, Loss_train: 0.132, Accuracy_train: 96.9 %, Loss_test 0.547, Accuracy_test: 82.2 %
    Epoch: 12, Loss_train: 0.225, Accuracy_train: 90.6 %, Loss_test 0.548, Accuracy_test: 82.0 %
    Epoch: 13, Loss_train: 0.122, Accuracy_train: 93.8 %, Loss_test 0.519, Accuracy_test: 82.0 %
    Epoch: 14, Loss_train: 0.299, Accuracy_train: 84.4 %, Loss_test 0.546, Accuracy_test: 82.9 %
    Epoch: 15, Loss_train: 0.205, Accuracy_train: 93.8 %, Loss_test 0.472, Accuracy_test: 85.3 %
    Epoch: 16, Loss_train: 0.090, Accuracy_train: 96.9 %, Loss_test 0.505, Accuracy_test: 84.0 %
    Epoch: 17, Loss_train: 0.308, Accuracy_train: 93.8 %, Loss_test 0.477, Accuracy_test: 85.9 %
    Epoch: 18, Loss_train: 0.251, Accuracy_train: 84.4 %, Loss_test 0.537, Accuracy_test: 84.5 %
    Epoch: 19, Loss_train: 0.095, Accuracy_train: 100.0 %, Loss_test 0.473, Accuracy_test: 85.1 %
    Epoch: 20, Loss_train: 0.281, Accuracy_train: 87.5 %, Loss_test 0.575, Accuracy_test: 83.9 %
    Epoch: 21, Loss_train: 0.079, Accuracy_train: 96.9 %, Loss_test 0.417, Accuracy_test: 85.9 %
    Epoch: 22, Loss_train: 0.043, Accuracy_train: 100.0 %, Loss_test 0.468, Accuracy_test: 86.1 %
    Epoch: 23, Loss_train: 0.236, Accuracy_train: 90.6 %, Loss_test 0.452, Accuracy_test: 86.6 %
    Epoch: 24, Loss_train: 0.074, Accuracy_train: 100.0 %, Loss_test 0.469, Accuracy_test: 85.7 %
    Epoch: 25, Loss_train: 0.081, Accuracy_train: 100.0 %, Loss_test 0.429, Accuracy_test: 86.3 %
    Epoch: 26, Loss_train: 0.041, Accuracy_train: 100.0 %, Loss_test 0.461, Accuracy_test: 87.2 %
    Epoch: 27, Loss_train: 0.254, Accuracy_train: 90.6 %, Loss_test 0.455, Accuracy_test: 86.1 %
    Epoch: 28, Loss_train: 0.077, Accuracy_train: 96.9 %, Loss_test 0.429, Accuracy_test: 87.1 %
    Epoch: 29, Loss_train: 0.106, Accuracy_train: 93.8 %, Loss_test 0.492, Accuracy_test: 84.1 %
    Epoch: 30, Loss_train: 0.068, Accuracy_train: 96.9 %, Loss_test 0.465, Accuracy_test: 85.3 %
    Epoch: 31, Loss_train: 0.108, Accuracy_train: 96.9 %, Loss_test 0.452, Accuracy_test: 88.1 %
    Epoch: 32, Loss_train: 0.114, Accuracy_train: 96.9 %, Loss_test 0.497, Accuracy_test: 86.4 %
    Epoch: 33, Loss_train: 0.078, Accuracy_train: 100.0 %, Loss_test 0.456, Accuracy_test: 87.6 %
    Epoch: 34, Loss_train: 0.025, Accuracy_train: 100.0 %, Loss_test 0.411, Accuracy_test: 87.0 %
    Epoch: 35, Loss_train: 0.180, Accuracy_train: 93.8 %, Loss_test 0.493, Accuracy_test: 86.8 %
    Epoch: 36, Loss_train: 0.066, Accuracy_train: 96.9 %, Loss_test 0.475, Accuracy_test: 85.8 %
    Epoch: 37, Loss_train: 0.168, Accuracy_train: 90.6 %, Loss_test 0.441, Accuracy_test: 86.7 %
    Epoch: 38, Loss_train: 0.157, Accuracy_train: 90.6 %, Loss_test 0.383, Accuracy_test: 88.7 %
    Epoch: 39, Loss_train: 0.166, Accuracy_train: 90.6 %, Loss_test 0.498, Accuracy_test: 85.2 %
    Epoch: 40, Loss_train: 0.179, Accuracy_train: 96.9 %, Loss_test 0.415, Accuracy_test: 87.7 %
    Epoch: 41, Loss_train: 0.140, Accuracy_train: 93.8 %, Loss_test 0.436, Accuracy_test: 87.5 %
    Epoch: 42, Loss_train: 0.116, Accuracy_train: 96.9 %, Loss_test 0.409, Accuracy_test: 89.0 %
    Epoch: 43, Loss_train: 0.104, Accuracy_train: 93.8 %, Loss_test 0.414, Accuracy_test: 87.6 %
    Epoch: 44, Loss_train: 0.060, Accuracy_train: 96.9 %, Loss_test 0.416, Accuracy_test: 88.0 %
    Epoch: 45, Loss_train: 0.090, Accuracy_train: 96.9 %, Loss_test 0.371, Accuracy_test: 89.6 %
    Epoch: 46, Loss_train: 0.025, Accuracy_train: 100.0 %, Loss_test 0.425, Accuracy_test: 88.9 %
    Epoch: 47, Loss_train: 0.065, Accuracy_train: 96.9 %, Loss_test 0.448, Accuracy_test: 87.6 %
    Epoch: 48, Loss_train: 0.049, Accuracy_train: 96.9 %, Loss_test 0.463, Accuracy_test: 87.7 %
    Epoch: 49, Loss_train: 0.057, Accuracy_train: 96.9 %, Loss_test 0.432, Accuracy_test: 87.0 %
    Epoch: 50, Loss_train: 0.078, Accuracy_train: 96.9 %, Loss_test 0.450, Accuracy_test: 87.4 %





    <matplotlib.legend.Legend at 0x7f0fd221e6d0>




    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_10_2.png)
    



```python
#PointFull, no augment
model = PointNetFull(NUM_POINTS,NUM_CLASSES)
(loss_train_array_POINTNetFull,
 accuracy_train_array_POINTNetFull,
 loss_test_array_POINTNetFull,
 accuracy_test_array_POINTNetFull) = train(
    model, 
    dataloader_train,
    dataloader_test, 
    epochs=50,
    loss_func=torch.nn.CrossEntropyLoss()
)

plt.figure(1)
plt.title("PointNetFull Acurracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")

plt.plot(accuracy_train_array_POINTNetFull,label="train_PointNetFull")
plt.plot(accuracy_test_array_POINTNetFull,label="test_PointNetFull")
plt.grid()
plt.legend()


```

    Epoch: 1, Loss_train: 2.265, Accuracy_train: 15.6 %, Loss_test 2.432, Accuracy_test: 14.2 %
    Epoch: 2, Loss_train: 0.911, Accuracy_train: 68.8 %, Loss_test 1.089, Accuracy_test: 63.5 %
    Epoch: 3, Loss_train: 0.590, Accuracy_train: 87.5 %, Loss_test 0.808, Accuracy_test: 73.9 %
    Epoch: 4, Loss_train: 0.284, Accuracy_train: 90.6 %, Loss_test 0.780, Accuracy_test: 75.8 %
    Epoch: 5, Loss_train: 0.210, Accuracy_train: 96.9 %, Loss_test 0.658, Accuracy_test: 78.8 %
    Epoch: 6, Loss_train: 0.501, Accuracy_train: 78.1 %, Loss_test 0.692, Accuracy_test: 77.4 %
    Epoch: 7, Loss_train: 0.363, Accuracy_train: 84.4 %, Loss_test 0.690, Accuracy_test: 76.7 %
    Epoch: 8, Loss_train: 0.192, Accuracy_train: 90.6 %, Loss_test 0.574, Accuracy_test: 82.3 %
    Epoch: 9, Loss_train: 0.391, Accuracy_train: 78.1 %, Loss_test 0.584, Accuracy_test: 81.8 %
    Epoch: 10, Loss_train: 0.374, Accuracy_train: 84.4 %, Loss_test 0.561, Accuracy_test: 82.6 %
    Epoch: 11, Loss_train: 0.289, Accuracy_train: 90.6 %, Loss_test 0.584, Accuracy_test: 80.2 %
    Epoch: 12, Loss_train: 0.356, Accuracy_train: 84.4 %, Loss_test 0.619, Accuracy_test: 81.2 %
    Epoch: 13, Loss_train: 0.070, Accuracy_train: 100.0 %, Loss_test 0.626, Accuracy_test: 80.6 %
    Epoch: 14, Loss_train: 0.059, Accuracy_train: 100.0 %, Loss_test 0.553, Accuracy_test: 83.9 %
    Epoch: 15, Loss_train: 0.211, Accuracy_train: 93.8 %, Loss_test 0.577, Accuracy_test: 81.9 %
    Epoch: 16, Loss_train: 0.270, Accuracy_train: 84.4 %, Loss_test 0.567, Accuracy_test: 81.5 %
    Epoch: 17, Loss_train: 0.215, Accuracy_train: 90.6 %, Loss_test 0.462, Accuracy_test: 85.3 %
    Epoch: 18, Loss_train: 0.164, Accuracy_train: 96.9 %, Loss_test 0.573, Accuracy_test: 82.3 %
    Epoch: 19, Loss_train: 0.205, Accuracy_train: 93.8 %, Loss_test 0.549, Accuracy_test: 83.3 %
    Epoch: 20, Loss_train: 0.297, Accuracy_train: 87.5 %, Loss_test 0.492, Accuracy_test: 84.7 %
    Epoch: 21, Loss_train: 0.092, Accuracy_train: 100.0 %, Loss_test 0.463, Accuracy_test: 87.4 %
    Epoch: 22, Loss_train: 0.220, Accuracy_train: 93.8 %, Loss_test 0.453, Accuracy_test: 85.8 %
    Epoch: 23, Loss_train: 0.213, Accuracy_train: 93.8 %, Loss_test 0.403, Accuracy_test: 87.2 %
    Epoch: 24, Loss_train: 0.108, Accuracy_train: 100.0 %, Loss_test 0.481, Accuracy_test: 86.2 %
    Epoch: 25, Loss_train: 0.104, Accuracy_train: 96.9 %, Loss_test 0.430, Accuracy_test: 85.8 %
    Epoch: 26, Loss_train: 0.335, Accuracy_train: 96.9 %, Loss_test 0.475, Accuracy_test: 87.3 %
    Epoch: 27, Loss_train: 0.090, Accuracy_train: 96.9 %, Loss_test 0.442, Accuracy_test: 86.7 %
    Epoch: 28, Loss_train: 0.022, Accuracy_train: 100.0 %, Loss_test 0.467, Accuracy_test: 86.2 %
    Epoch: 29, Loss_train: 0.060, Accuracy_train: 96.9 %, Loss_test 0.529, Accuracy_test: 83.3 %
    Epoch: 30, Loss_train: 0.040, Accuracy_train: 100.0 %, Loss_test 0.413, Accuracy_test: 87.6 %
    Epoch: 31, Loss_train: 0.210, Accuracy_train: 87.5 %, Loss_test 0.470, Accuracy_test: 85.5 %
    Epoch: 32, Loss_train: 0.216, Accuracy_train: 93.8 %, Loss_test 0.473, Accuracy_test: 85.2 %
    Epoch: 33, Loss_train: 0.125, Accuracy_train: 93.8 %, Loss_test 0.489, Accuracy_test: 85.3 %
    Epoch: 34, Loss_train: 0.176, Accuracy_train: 93.8 %, Loss_test 0.448, Accuracy_test: 88.7 %
    Epoch: 35, Loss_train: 0.188, Accuracy_train: 90.6 %, Loss_test 0.445, Accuracy_test: 87.1 %
    Epoch: 36, Loss_train: 0.090, Accuracy_train: 96.9 %, Loss_test 0.445, Accuracy_test: 86.4 %
    Epoch: 37, Loss_train: 0.132, Accuracy_train: 96.9 %, Loss_test 0.507, Accuracy_test: 86.6 %
    Epoch: 38, Loss_train: 0.327, Accuracy_train: 84.4 %, Loss_test 0.444, Accuracy_test: 86.7 %
    Epoch: 39, Loss_train: 0.252, Accuracy_train: 93.8 %, Loss_test 0.482, Accuracy_test: 87.1 %
    Epoch: 40, Loss_train: 0.120, Accuracy_train: 93.8 %, Loss_test 0.528, Accuracy_test: 84.6 %
    Epoch: 41, Loss_train: 0.043, Accuracy_train: 100.0 %, Loss_test 0.449, Accuracy_test: 87.2 %
    Epoch: 42, Loss_train: 0.070, Accuracy_train: 96.9 %, Loss_test 0.443, Accuracy_test: 87.9 %
    Epoch: 43, Loss_train: 0.179, Accuracy_train: 93.8 %, Loss_test 0.451, Accuracy_test: 86.9 %
    Epoch: 44, Loss_train: 0.032, Accuracy_train: 100.0 %, Loss_test 0.491, Accuracy_test: 85.7 %
    Epoch: 45, Loss_train: 0.042, Accuracy_train: 96.9 %, Loss_test 0.432, Accuracy_test: 88.3 %
    Epoch: 46, Loss_train: 0.029, Accuracy_train: 100.0 %, Loss_test 0.452, Accuracy_test: 87.4 %
    Epoch: 47, Loss_train: 0.136, Accuracy_train: 96.9 %, Loss_test 0.517, Accuracy_test: 87.0 %
    Epoch: 48, Loss_train: 0.091, Accuracy_train: 96.9 %, Loss_test 0.420, Accuracy_test: 87.8 %
    Epoch: 49, Loss_train: 0.097, Accuracy_train: 96.9 %, Loss_test 0.449, Accuracy_test: 86.0 %
    Epoch: 50, Loss_train: 0.047, Accuracy_train: 100.0 %, Loss_test 0.479, Accuracy_test: 86.7 %





    <matplotlib.legend.Legend at 0x7f0fd06614c0>




    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_11_2.png)
    



```python
plt.figure(1)
plt.title("PointNetBasic vs PointNetFull Acurracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")
plt.plot(accuracy_train_array_POINTNetBasic,label="train_PointNetBasic")
plt.plot(accuracy_test_array_POINTNetBasic,label="test_PointNetBasic")
plt.plot(accuracy_train_array_POINTNetFull,label="train_PointNetFull")
plt.plot(accuracy_test_array_POINTNetFull,label="test_PointNetFull")
plt.grid()
plt.legend()


```




    <matplotlib.legend.Legend at 0x7f0fd0658af0>




    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_12_1.png)
    



```python

#PointFull, augment, AxisReducing
model = PointNetBasic(NUM_POINTS,NUM_CLASSES)
(loss_train_array_POINTNetBasic_augment,
 accuracy_train_array_POINTNetBasic_augment,
 loss_test_array_POINTNetBasic_augment,
 accuracy_test_array_POINTNetBasic_augment) = train(
    model, 
    dataloader_train_augment,
    dataloader_test_augment, 
    epochs=50,
    loss_func=torch.nn.CrossEntropyLoss()
)

plt.figure(1)
plt.title("PointNetBasic_augment Acurracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")

plt.plot(accuracy_train_array_POINTNetBasic,label="train_PointNetBasic")
plt.plot(accuracy_test_array_POINTNetBasic,label="test_PointNetBasic")

plt.plot(accuracy_train_array_POINTNetBasic_augment,label="train_PointNetBasic_augment")
plt.plot(accuracy_test_array_POINTNetBasic_augment,label="test_PointNetBasic_augment")
plt.grid()
plt.legend()


```

    Epoch: 1, Loss_train: 2.526, Accuracy_train: 3.1 %, Loss_test 2.044, Accuracy_test: 29.7 %
    Epoch: 2, Loss_train: 0.319, Accuracy_train: 90.6 %, Loss_test 0.733, Accuracy_test: 76.8 %
    Epoch: 3, Loss_train: 0.454, Accuracy_train: 84.4 %, Loss_test 0.705, Accuracy_test: 78.8 %
    Epoch: 4, Loss_train: 0.259, Accuracy_train: 93.8 %, Loss_test 0.665, Accuracy_test: 77.5 %
    Epoch: 5, Loss_train: 0.255, Accuracy_train: 87.5 %, Loss_test 0.731, Accuracy_test: 75.3 %
    Epoch: 6, Loss_train: 0.225, Accuracy_train: 93.8 %, Loss_test 0.647, Accuracy_test: 79.3 %
    Epoch: 7, Loss_train: 0.645, Accuracy_train: 75.0 %, Loss_test 0.645, Accuracy_test: 79.2 %
    Epoch: 8, Loss_train: 0.250, Accuracy_train: 87.5 %, Loss_test 0.621, Accuracy_test: 80.8 %
    Epoch: 9, Loss_train: 0.208, Accuracy_train: 96.9 %, Loss_test 0.582, Accuracy_test: 80.7 %
    Epoch: 10, Loss_train: 0.122, Accuracy_train: 93.8 %, Loss_test 0.600, Accuracy_test: 81.0 %
    Epoch: 11, Loss_train: 0.298, Accuracy_train: 93.8 %, Loss_test 0.623, Accuracy_test: 81.3 %
    Epoch: 12, Loss_train: 0.112, Accuracy_train: 96.9 %, Loss_test 0.664, Accuracy_test: 80.7 %
    Epoch: 13, Loss_train: 0.281, Accuracy_train: 87.5 %, Loss_test 0.617, Accuracy_test: 79.6 %
    Epoch: 14, Loss_train: 0.069, Accuracy_train: 100.0 %, Loss_test 0.583, Accuracy_test: 81.9 %
    Epoch: 15, Loss_train: 0.106, Accuracy_train: 96.9 %, Loss_test 0.608, Accuracy_test: 82.2 %
    Epoch: 16, Loss_train: 0.173, Accuracy_train: 96.9 %, Loss_test 0.569, Accuracy_test: 81.7 %
    Epoch: 17, Loss_train: 0.181, Accuracy_train: 93.8 %, Loss_test 0.595, Accuracy_test: 82.6 %
    Epoch: 18, Loss_train: 0.156, Accuracy_train: 93.8 %, Loss_test 0.592, Accuracy_test: 80.1 %
    Epoch: 19, Loss_train: 0.325, Accuracy_train: 87.5 %, Loss_test 0.589, Accuracy_test: 82.8 %
    Epoch: 20, Loss_train: 0.186, Accuracy_train: 90.6 %, Loss_test 0.572, Accuracy_test: 83.6 %
    Epoch: 21, Loss_train: 0.079, Accuracy_train: 96.9 %, Loss_test 0.566, Accuracy_test: 83.4 %
    Epoch: 22, Loss_train: 0.300, Accuracy_train: 87.5 %, Loss_test 0.567, Accuracy_test: 82.5 %
    Epoch: 23, Loss_train: 0.023, Accuracy_train: 100.0 %, Loss_test 0.554, Accuracy_test: 84.4 %
    Epoch: 24, Loss_train: 0.230, Accuracy_train: 93.8 %, Loss_test 0.538, Accuracy_test: 83.3 %
    Epoch: 25, Loss_train: 0.040, Accuracy_train: 100.0 %, Loss_test 0.516, Accuracy_test: 85.2 %
    Epoch: 26, Loss_train: 0.079, Accuracy_train: 93.8 %, Loss_test 0.571, Accuracy_test: 84.2 %
    Epoch: 27, Loss_train: 0.091, Accuracy_train: 96.9 %, Loss_test 0.511, Accuracy_test: 85.4 %
    Epoch: 28, Loss_train: 0.079, Accuracy_train: 100.0 %, Loss_test 0.634, Accuracy_test: 83.8 %
    Epoch: 29, Loss_train: 0.122, Accuracy_train: 93.8 %, Loss_test 0.635, Accuracy_test: 82.1 %
    Epoch: 30, Loss_train: 0.084, Accuracy_train: 96.9 %, Loss_test 0.583, Accuracy_test: 83.7 %
    Epoch: 31, Loss_train: 0.262, Accuracy_train: 87.5 %, Loss_test 0.639, Accuracy_test: 83.5 %
    Epoch: 32, Loss_train: 0.146, Accuracy_train: 93.8 %, Loss_test 0.648, Accuracy_test: 81.4 %
    Epoch: 33, Loss_train: 0.133, Accuracy_train: 93.8 %, Loss_test 0.699, Accuracy_test: 82.6 %
    Epoch: 34, Loss_train: 0.141, Accuracy_train: 93.8 %, Loss_test 0.627, Accuracy_test: 83.2 %
    Epoch: 35, Loss_train: 0.025, Accuracy_train: 100.0 %, Loss_test 0.653, Accuracy_test: 84.1 %
    Epoch: 36, Loss_train: 0.082, Accuracy_train: 96.9 %, Loss_test 0.589, Accuracy_test: 85.0 %
    Epoch: 37, Loss_train: 0.077, Accuracy_train: 96.9 %, Loss_test 0.632, Accuracy_test: 84.2 %
    Epoch: 38, Loss_train: 0.133, Accuracy_train: 96.9 %, Loss_test 0.577, Accuracy_test: 85.8 %
    Epoch: 39, Loss_train: 0.027, Accuracy_train: 100.0 %, Loss_test 0.565, Accuracy_test: 85.6 %
    Epoch: 40, Loss_train: 0.048, Accuracy_train: 100.0 %, Loss_test 0.619, Accuracy_test: 84.3 %
    Epoch: 41, Loss_train: 0.116, Accuracy_train: 96.9 %, Loss_test 0.601, Accuracy_test: 84.1 %
    Epoch: 42, Loss_train: 0.006, Accuracy_train: 100.0 %, Loss_test 0.633, Accuracy_test: 84.7 %
    Epoch: 43, Loss_train: 0.013, Accuracy_train: 100.0 %, Loss_test 0.678, Accuracy_test: 84.4 %
    Epoch: 44, Loss_train: 0.029, Accuracy_train: 100.0 %, Loss_test 0.532, Accuracy_test: 85.6 %
    Epoch: 45, Loss_train: 0.049, Accuracy_train: 100.0 %, Loss_test 0.627, Accuracy_test: 85.2 %
    Epoch: 46, Loss_train: 0.079, Accuracy_train: 96.9 %, Loss_test 0.671, Accuracy_test: 84.4 %
    Epoch: 47, Loss_train: 0.065, Accuracy_train: 96.9 %, Loss_test 0.643, Accuracy_test: 85.0 %
    Epoch: 48, Loss_train: 0.012, Accuracy_train: 100.0 %, Loss_test 0.651, Accuracy_test: 85.1 %
    Epoch: 49, Loss_train: 0.010, Accuracy_train: 100.0 %, Loss_test 0.658, Accuracy_test: 83.4 %
    Epoch: 50, Loss_train: 0.054, Accuracy_train: 96.9 %, Loss_test 0.669, Accuracy_test: 84.7 %





    <matplotlib.legend.Legend at 0x7f0fd056d910>




    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_13_2.png)
    



```python
#PointNetBasic, augment, voxel
model = PointNetBasic(NUM_POINTS,NUM_CLASSES)
(loss_train_array_POINTNetBasic_augment_voxel,
 accuracy_train_array_POINTNetBasic_augment_voxel,
 loss_test_array_POINTNetBasic_augment_voxel,
 accuracy_test_array_POINTNetBasic_augment_voxel) = train(
    model, 
    dataloader_train_augment_voxel,
    dataloader_test_augment_voxel, 
    epochs=50,
    loss_func=torch.nn.CrossEntropyLoss()
)

plt.figure(1)
plt.title("PointNetBasic_augment_voxel Acurracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy in %")

plt.plot(accuracy_train_array_POINTNetBasic,label="train_PointNetBasic")
plt.plot(accuracy_test_array_POINTNetBasic,label="test_PointNetBasic")

plt.plot(accuracy_train_array_POINTNetBasic_augment_voxel,label="train_PointNetBasic_augment_voxel")
plt.plot(accuracy_test_array_POINTNetBasic_augment_voxel,label="test_PointNetBasic_augment_voxel")
plt.grid()
plt.legend()

```

    Epoch: 1, Loss_train: 2.389, Accuracy_train: 12.5 %, Loss_test 2.181, Accuracy_test: 23.2 %
    Epoch: 2, Loss_train: 0.647, Accuracy_train: 87.5 %, Loss_test 0.883, Accuracy_test: 70.0 %
    Epoch: 3, Loss_train: 0.401, Accuracy_train: 90.6 %, Loss_test 0.782, Accuracy_test: 72.7 %
    Epoch: 4, Loss_train: 0.564, Accuracy_train: 84.4 %, Loss_test 0.662, Accuracy_test: 77.3 %
    Epoch: 5, Loss_train: 0.404, Accuracy_train: 81.2 %, Loss_test 0.727, Accuracy_test: 76.4 %
    Epoch: 6, Loss_train: 0.397, Accuracy_train: 87.5 %, Loss_test 0.654, Accuracy_test: 77.2 %
    Epoch: 7, Loss_train: 0.241, Accuracy_train: 93.8 %, Loss_test 0.664, Accuracy_test: 78.2 %
    Epoch: 8, Loss_train: 0.390, Accuracy_train: 84.4 %, Loss_test 0.621, Accuracy_test: 79.4 %
    Epoch: 9, Loss_train: 0.430, Accuracy_train: 90.6 %, Loss_test 0.694, Accuracy_test: 76.9 %
    Epoch: 10, Loss_train: 0.392, Accuracy_train: 81.2 %, Loss_test 0.567, Accuracy_test: 79.0 %
    Epoch: 11, Loss_train: 0.365, Accuracy_train: 90.6 %, Loss_test 0.542, Accuracy_test: 82.7 %
    Epoch: 12, Loss_train: 0.208, Accuracy_train: 96.9 %, Loss_test 0.566, Accuracy_test: 81.4 %
    Epoch: 13, Loss_train: 0.192, Accuracy_train: 93.8 %, Loss_test 0.562, Accuracy_test: 81.1 %
    Epoch: 14, Loss_train: 0.452, Accuracy_train: 81.2 %, Loss_test 0.551, Accuracy_test: 80.9 %
    Epoch: 15, Loss_train: 0.294, Accuracy_train: 87.5 %, Loss_test 0.613, Accuracy_test: 80.9 %
    Epoch: 16, Loss_train: 0.186, Accuracy_train: 93.8 %, Loss_test 0.587, Accuracy_test: 81.1 %
    Epoch: 17, Loss_train: 0.268, Accuracy_train: 90.6 %, Loss_test 0.511, Accuracy_test: 84.2 %
    Epoch: 18, Loss_train: 0.100, Accuracy_train: 100.0 %, Loss_test 0.577, Accuracy_test: 82.3 %
    Epoch: 19, Loss_train: 0.253, Accuracy_train: 93.8 %, Loss_test 0.526, Accuracy_test: 83.2 %
    Epoch: 20, Loss_train: 0.220, Accuracy_train: 93.8 %, Loss_test 0.579, Accuracy_test: 81.9 %
    Epoch: 21, Loss_train: 0.099, Accuracy_train: 96.9 %, Loss_test 0.497, Accuracy_test: 84.4 %
    Epoch: 22, Loss_train: 0.174, Accuracy_train: 93.8 %, Loss_test 0.512, Accuracy_test: 83.4 %
    Epoch: 23, Loss_train: 0.500, Accuracy_train: 87.5 %, Loss_test 0.496, Accuracy_test: 84.1 %
    Epoch: 24, Loss_train: 0.272, Accuracy_train: 84.4 %, Loss_test 0.444, Accuracy_test: 85.8 %
    Epoch: 25, Loss_train: 0.240, Accuracy_train: 87.5 %, Loss_test 0.533, Accuracy_test: 83.4 %
    Epoch: 26, Loss_train: 0.066, Accuracy_train: 100.0 %, Loss_test 0.470, Accuracy_test: 87.4 %
    Epoch: 27, Loss_train: 0.134, Accuracy_train: 96.9 %, Loss_test 0.475, Accuracy_test: 85.0 %
    Epoch: 28, Loss_train: 0.355, Accuracy_train: 90.6 %, Loss_test 0.490, Accuracy_test: 85.3 %
    Epoch: 29, Loss_train: 0.148, Accuracy_train: 96.9 %, Loss_test 0.455, Accuracy_test: 85.9 %
    Epoch: 30, Loss_train: 0.273, Accuracy_train: 90.6 %, Loss_test 0.489, Accuracy_test: 86.1 %
    Epoch: 31, Loss_train: 0.200, Accuracy_train: 93.8 %, Loss_test 0.432, Accuracy_test: 87.2 %
    Epoch: 32, Loss_train: 0.122, Accuracy_train: 93.8 %, Loss_test 0.477, Accuracy_test: 85.4 %
    Epoch: 33, Loss_train: 0.152, Accuracy_train: 96.9 %, Loss_test 0.454, Accuracy_test: 84.8 %
    Epoch: 34, Loss_train: 0.410, Accuracy_train: 84.4 %, Loss_test 0.523, Accuracy_test: 84.1 %
    Epoch: 35, Loss_train: 0.218, Accuracy_train: 90.6 %, Loss_test 0.488, Accuracy_test: 85.6 %
    Epoch: 36, Loss_train: 0.311, Accuracy_train: 90.6 %, Loss_test 0.479, Accuracy_test: 86.1 %
    Epoch: 37, Loss_train: 0.145, Accuracy_train: 90.6 %, Loss_test 0.460, Accuracy_test: 86.1 %
    Epoch: 38, Loss_train: 0.453, Accuracy_train: 81.2 %, Loss_test 0.463, Accuracy_test: 85.5 %
    Epoch: 39, Loss_train: 0.260, Accuracy_train: 87.5 %, Loss_test 0.470, Accuracy_test: 85.6 %
    Epoch: 40, Loss_train: 0.563, Accuracy_train: 87.5 %, Loss_test 0.516, Accuracy_test: 84.6 %
    Epoch: 41, Loss_train: 0.258, Accuracy_train: 87.5 %, Loss_test 0.434, Accuracy_test: 86.8 %
    Epoch: 42, Loss_train: 0.214, Accuracy_train: 87.5 %, Loss_test 0.438, Accuracy_test: 86.6 %
    Epoch: 43, Loss_train: 0.180, Accuracy_train: 93.8 %, Loss_test 0.417, Accuracy_test: 87.3 %
    Epoch: 44, Loss_train: 0.114, Accuracy_train: 93.8 %, Loss_test 0.426, Accuracy_test: 86.4 %
    Epoch: 45, Loss_train: 0.207, Accuracy_train: 96.9 %, Loss_test 0.429, Accuracy_test: 85.9 %
    Epoch: 46, Loss_train: 0.250, Accuracy_train: 87.5 %, Loss_test 0.427, Accuracy_test: 87.6 %
    Epoch: 47, Loss_train: 0.186, Accuracy_train: 93.8 %, Loss_test 0.465, Accuracy_test: 85.9 %
    Epoch: 48, Loss_train: 0.172, Accuracy_train: 93.8 %, Loss_test 0.433, Accuracy_test: 85.7 %
    Epoch: 49, Loss_train: 0.073, Accuracy_train: 96.9 %, Loss_test 0.464, Accuracy_test: 86.5 %
    Epoch: 50, Loss_train: 0.142, Accuracy_train: 93.8 %, Loss_test 0.467, Accuracy_test: 86.9 %





    <matplotlib.legend.Legend at 0x7f0fd0510fd0>




    
![png](main_CHUPIN_VILLENEUVE_files/main_CHUPIN_VILLENEUVE_14_2.png)
    


# The model :


```python
model = PointMLP(NUM_POINTS,NUM_CLASSES)
print(model)
```

    PointMLP(
      (fc_1): Linear(in_features=3072, out_features=512, bias=True)
      (bn_1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_2): Linear(in_features=512, out_features=256, bias=True)
      (dropout_1): Dropout(p=0.3, inplace=False)
      (bn_2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_3): Linear(in_features=256, out_features=10, bias=True)
      (bn_3): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )



```python
model = PointNetBasic(NUM_POINTS,NUM_CLASSES)
print(model)
```

    PointNetBasic(
      (fc_1): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
      (bn_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_2): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      (bn_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      (bn_3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_4): Conv1d(32, 128, kernel_size=(1,), stride=(1,))
      (bn_4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_5): Conv1d(128, 512, kernel_size=(1,), stride=(1,))
      (bn_5): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (mp): MaxPool1d(kernel_size=512, stride=512, padding=0, dilation=1, ceil_mode=False)
      (fc_6): Linear(in_features=1024, out_features=256, bias=True)
      (bn_6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_7): Linear(in_features=256, out_features=128, bias=True)
      (dropout_1): Dropout(p=0.3, inplace=False)
      (bn_7): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_8): Linear(in_features=128, out_features=10, bias=True)
    )



```python
model = PointNetFull(NUM_POINTS,NUM_CLASSES)
print(model)
```

    PointNetFull(
      (input_transform_1): InputTransform(
        (t_net): Tnet(
          (fc_1): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
          (bn_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (fc_2): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
          (bn_2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (fc_3): Conv1d(64, 256, kernel_size=(1,), stride=(1,))
          (bn_3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (mp): MaxPool1d(kernel_size=256, stride=256, padding=0, dilation=1, ceil_mode=False)
          (fc_4): Linear(in_features=1024, out_features=128, bias=True)
          (bn_4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (fc_5): Linear(in_features=128, out_features=64, bias=True)
          (bn_5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (fc_6): Linear(in_features=64, out_features=9, bias=True)
        )
      )
      (fc_1): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
      (bn_1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_2): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      (bn_2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_3): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      (bn_3): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_4): Conv1d(32, 128, kernel_size=(1,), stride=(1,))
      (bn_4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_5): Conv1d(128, 512, kernel_size=(1,), stride=(1,))
      (bn_5): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (mp): MaxPool1d(kernel_size=512, stride=512, padding=0, dilation=1, ceil_mode=False)
      (fc_6): Linear(in_features=1024, out_features=256, bias=True)
      (bn_6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_7): Linear(in_features=256, out_features=128, bias=True)
      (dropout_1): Dropout(p=0.3, inplace=False)
      (bn_7): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc_8): Linear(in_features=128, out_features=10, bias=True)
    )
