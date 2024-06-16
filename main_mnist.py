import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

import random, os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import pandas as pd
import copy


from fedlab.utils.dataset import FMNISTPartitioner,CIFAR10Partitioner, MNISTPartitioner
from fedlab.utils.functional import partition_report, save_dict
from generate_synthetic_mnist import SyntheticMnistDataset
    
from args_mnist import args_parser
import server_se1 as server
import model

from utils.global_test import test_on_globaldataset, globalmodel_test_on_localdataset,globalmodel_test_on_specifdataset
from utils.local_test import test_on_localdataset
from utils.training_loss import train_loss_show,train_localacc_show, train_globalacc_show
from utils.sampling import testset_sampling, trainset_sampling, trainset_sampling_label
# from utils.clusteror import cluster_clients
from utils.tSNE import FeatureVisualize
import selective_he

args = args_parser()


def seed_torch(seed=args.seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)  
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()
GLOBAL_SEED = 1
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

similarity = False
save_models = False
Train_model = True

C = "2CNN_2"

trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
root = "data/mnist/"

# trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=trans_mnist)
# testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=trans_mnist)

trainset = datasets.MNIST(root=root, train=True, download=True, transform=trans_mnist)
testset = datasets.MNIST(root=root, train=False, download=True, transform=trans_mnist)

num_classes = args.num_classes
num_clients = args.K
number_perclass = args.num_perclass
 

col_names = [f"class{i}" for i in range(num_classes)]
print(col_names)
hist_color = '#4169E1'
plt.rcParams['figure.facecolor'] = 'white'


# perform partition
noniid_labeldir_part = MNISTPartitioner(trainset.targets, 
                                    num_clients=num_clients,
                                    partition="noniid-#label",
                                    major_classes_num=1,
                                    seed=1)


# generate partition report
csv_file = "data/mnist/mnist_noniid_labeldir_clients_10.csv"
partition_report(trainset.targets, noniid_labeldir_part.client_dict, 
                 class_num=num_classes, 
                 verbose=False, file=csv_file)

noniid_labeldir_part_df = pd.read_csv(csv_file,header=1)
noniid_labeldir_part_df = noniid_labeldir_part_df.set_index('client')
for col in col_names:
    noniid_labeldir_part_df[col] = (noniid_labeldir_part_df[col] * noniid_labeldir_part_df['Amount']).astype(int)

# select first 10 clients for bar plot
noniid_labeldir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
# plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"data/mnist//mnist_noniid_labeldir_clients_10.png", 
            dpi=400, bbox_inches = 'tight')

# split dataset into training and testing

# # perform partition
# iid_part = MNISTPartitioner(trainset.targets, 
#                             num_clients=num_clients,
#                             partition="iid",
#                             seed=1)

# # generate partition report
# csv_file = "data/mnist/mnist_iid_clients_10.csv"
# partition_report(trainset.targets, iid_part.client_dict, 
#                  class_num=num_classes, 
#                  verbose=False, file=csv_file)

# iid_part_df = pd.read_csv(csv_file,header=1)
# iid_part_df = iid_part_df.set_index('client')
# for col in col_names:
#     iid_part_df[col] = (iid_part_df[col] * iid_part_df['Amount']).astype(int)

# # select first 10 clients for bar plot
# iid_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
# # plt.tight_layout()
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.xlabel('sample num')
# plt.savefig(f"data/mnist/mnist_iid_clients_10.png", 
#             dpi=400, bbox_inches = 'tight')



trainset_sample_rate = args.trainset_sample_rate
rare_class_nums = 0
dict_users_train = trainset_sampling_label(args, trainset, trainset_sample_rate,rare_class_nums, noniid_labeldir_part) 
dict_users_test = testset_sampling(args, testset, number_perclass, noniid_labeldir_part_df)


# dict_users_train_iid = trainset_sampling_label(args, trainset, trainset_sample_rate,rare_class_nums, iid_part) 
# dict_users_test_iid = testset_sampling(args, testset, number_perclass, iid_part_df)

# Generate synthetic dataset
synthetic_dataset = SyntheticMnistDataset(num_classes=num_classes, train = True)
synthetic_noniid_labeldir_part = MNISTPartitioner(synthetic_dataset.targets, 
                                num_clients=num_clients,
                                partition="noniid-#label",
                                major_classes_num=1,
                                seed=1)
synthetic_dict_users = trainset_sampling_label(args, synthetic_dataset, trainset_sample_rate,rare_class_nums, synthetic_noniid_labeldir_part)

specf_model = model.Client_Model(args, name='mnist').to(args.device)

serverz = server.Server(args, specf_model, trainset, synthetic_dataset.dataset_split, dict_users_train, synthetic_dict_users)#dict_users指的是user的local dataset索引
print("global_model: ", serverz.nn.state_dict)

def run_FedAvg():
    print("running FedAvg")
    print("total clients: ", args.k)
    print("active clients: ", args.C * args.K)

    server_iid = server.Server(args, specf_model, trainset, dict_users_train_iid)
    if Train_model:
        global_model_iid, similarity_dict_iid, client_models_iid, loss_dict_iid, clients_index_iid, acc_list_iid = server_iid.fedavg_joint_update(testset, dict_users_test_iid[0], iid= True, similarity = similarity, test_global_model_accuracy = True)
    else:
        if similarity:
            similarity_dict_iid = torch.load("results/Test/label skew/mnist/iid-fedavg/seed{}/similarity_dict_iid_{}E_{}class.pt".format(args.seed,args.E,C))
        acc_list_iid = torch.load("results/Test/label skew/mnist/iid-fedavg/seed{}/acc_list_iid_{}E_{}class.pt".format(args.seed,args.E,C))
        global_model_iid = server_iid.nn
        client_models_iid = server_iid.nns
        path_iid_fedavg = "results/Test/label skew/mnist/iid-fedavg/seed{}/global_model_iid-fedavg_{}E_{}class.pt".format(args.seed,args.E,C)
        global_model_iid.load_state_dict(torch.load(path_iid_fedavg))
        for i in range(args.K):
            path_iid_fedavg = "results/Test/label skew/mnist/iid-fedavg/seed{}/client{}_model_fedavg_{}E_{}class".format(args.seed,i,args.E,C)
            client_models_iid[i]=copy.deepcopy(global_model_iid)
            client_models_iid[i].load_state_dict(torch.load(path_iid_fedavg))

def run_FedFA():
    init_model = copy.deepcopy(serverz)
    encryption_mask = selective_he.calculate_mask(args, init_model, trainset)
    server_feature = copy.deepcopy(serverz)
    if Train_model:
        global_modelfa, similarity_dictfa, client_modelsfa, loss_dictfa, clients_indexfa, acc_listfa = server_feature.fedfa_anchorloss(testset, dict_users_test[0], encryption_mask, similarity = similarity, test_global_model_accuracy = True)
    else:
        if similarity:
            similarity_dictfa = torch.load("results/Test/label skew/mnist/fedfa/seed{}/similarity_dictfa_{}E_{}class.pt".format(args.seed,args.E,C))
        acc_listfa = torch.load("results/Test/label skew/mnist/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed,args.E,C))
        global_modelfa = server_feature.nn
        client_modelsfa = server_feature.nns
        path_fedfa = "results/Test/label skew/mnist/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed,args.E,C)
        global_modelfa.load_state_dict(torch.load(path_fedfa))
        for i in range(args.K):
            path_fedfa = "results/Test/label skew/mnist/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(args.seed,i,args.E,C)
            client_modelsfa[i] = copy.deepcopy(global_modelfa)
            client_modelsfa[i].load_state_dict(torch.load(path_fedfa))
    
    if save_models:
        if similarity:
            torch.save(similarity_dictfa,"results/Test/label skew/mnist/iid-fedavg/seed{}/similarity_dictfa_{}E_{}class.pt".format(args.seed,args.E,C))
    torch.save(acc_listfa,"results/Test/label skew/mnist/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(args.seed,args.E,C))
    path_fedfa = "results/Test/label skew/mnist/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(args.seed,args.E,C)
    torch.save(global_modelfa.state_dict(), path_fedfa)

    if Train_model:
        #train_loss_show(args, loss_dictfa, clients_indexfa)
        train_globalacc_show(args, acc_listfa)

if __name__ == "__main__":
    print(args.dataset)
    print("setting: ", int(args.C*args.K), "/", args.K)
    # run_FedDyn()
    run_FedFA()
    # run_FedAvg()