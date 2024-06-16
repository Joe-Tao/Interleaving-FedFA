import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils.optimizer as op
import tenseal as ts
import utils.ckks as ckks
import threading
import queue
import time

def calculate_sensitivity_map(args, global_model, dataset, dataset_index):
    sensitivity_maps = []
    
    for k in range(args.K):
        Dtr = DataLoader(op.DatasetSplit(dataset, dataset_index[k]), batch_size=args.B, shuffle=True)
        loss_function = torch.nn.CrossEntropyLoss().to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = global_model(imgs)

        loss = loss_function(y_preds, labels)
        loss.backward()
        gradients = [torch.abs(param.grad.data) for param in global_model.parameters()]
        sensitivity_maps.append(gradients)

    return sensitivity_maps

def client_fedfa_cl(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict, masks, global_list, clients_list, shape_list, isAuthentic):  # update nn
    if isAuthentic:
        # Decryption
        if global_round == 0:
            for k in client_index: #k is the index of the client
                client_models[k] = copy.deepcopy(global_model)
                for p, layer in zip(client_models[k].parameters(), clients_list[k]):
                    p.data.copy_(layer)
                anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
                loss_dict[k].extend(loss)
        else:
            decryption_threads = []
            decryption_queue = queue.Queue()
            for k in client_index:
                thread = threading.Thread(target=decrypt_by_para_v2, args=(args, client_models, clients_list[k], shape_list, k, global_model, decryption_queue))
                decryption_threads.append(thread)
                thread.start()
                
            for thread in decryption_threads:
                thread.join()
            
            # while not decryption_queue.empty():
            #     k, client_model = decryption_queue.get()
            #     client_models[k] = client_model
        
            # Optimize
            for k in client_index:
                anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
                loss_dict[k].extend(loss)
        
        # Encryption
        encryption_threads = []
        encryption_queue = queue.Queue()
        for k in client_index:
            thread = threading.Thread(target=encrypt_by_para, args=(args, clients_list, client_models[k], k, masks, encryption_queue))
            encryption_threads.append(thread)
            thread.start()
            
        for thread in encryption_threads:
            thread.join()
        # while not encryption_queue.empty():
        #     k, temp_layer_list = encryption_queue.get()
        #     clients_list[k] = temp_layer_list
            #print(k, clients_list[k])
            # temp = encrypt_by_para(args, client_model,  masks)
            # clients_list[k] = temp
            
       
    else:
        for k in client_index:
            client_models[k] = copy.deepcopy(global_model)
            for p, layer in zip(client_models[k].parameters(), clients_list[k]):
                p.data.copy_(layer)
            
            anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k])
    
            loss_dict[k].extend(loss)
            
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss) 
     
        
    return anchorloss_funcs, client_models, loss_dict, clients_list



def model_to_list(args, client_model):
    temp = []
    for layer in client_model.parameters():
        temp.append(layer)
    return temp


    

def list_to_model(args, model_list, model):
    for p, layer in zip(model.parameters(), model_list):
        layer = layer.to(dtype=p.data.dtype, device=p.data.device)
        p.data.copy_(layer)
    return model




def encrypt_by_para_v3(args, client_model, masks):
    state_dict = client_model.state_dict()
    encrypted_state_dict = {}
    layer_count = 0
    for key, tensor in state_dict.items():
        layer_count += 1
        if layer_count in [2, 4, 6]:
            flat_tensor = tensor.flatten()
            mask_layer = masks[layer_count-1].flatten()
            encrypted_tensor = [
                ckks.EncryptionManager().encrypt_vector(flat_tensor[i].item())
                if mask_layer[i] == 1 else flat_tensor[i].item()
                for i in range(flat_tensor.numel())
            ]
            encrypted_state_dict[key] = encrypted_tensor
        else:
            encrypted_state_dict[key] = tensor

    return encrypted_state_dict



def decryption_to_plain(layer):
    ckks_count = 0
    if isinstance(layer, list):
        return [decryption_to_plain(sublayer) for sublayer in layer]
    elif isinstance(layer, ts.CKKSVector):
        ckks_count += 1
        # print("CKKSVector")
        return layer.decrypt()[0]
    else:
        # print("count of CKKSVector: ", ckks_count)
        return layer
    

def decrypt_by_para_v2(args, client_models, layer_list, shape_list, k, model, decryption_queue):

    # temp = copy.deepcopy(model)        
    decrypted_layers = []
    for idx, (layer_encrypted, shape) in enumerate(zip(layer_list, shape_list)):
        # Decrypt all layers uniformly
        decrypted_layer = decryption_to_plain(layer_encrypted) if idx in [1, 3, 4, 5] else layer_encrypted
        tensor = torch.Tensor(decrypted_layer).view(shape).to(args.device)
        decrypted_layers.append(tensor)

    with torch.no_grad():  # Ensure we do not track these operations in the gradient computation
        for param, new_data in zip(model.parameters(), decrypted_layers):
            param.data.copy_(new_data)  # Use copy_ to maintain correct device and data types
            # param.data.copy_(torch.zeros_like(param.data))
    
    client_models[k] = model
    #decryption_queue.put((k, model))
    #return model

def encrypt_by_para(args, clients_list, client_model, k, masks, encryption_queue):
    layer_count = 0
    # temp = []
    for layer, mask_layer in zip(client_model.parameters(), masks):
        layer_count += 1
        # print("layer size: ", layer.size())
        if (layer_count in [2, 4, 5, 6] ):
            flat_tensor = layer.flatten()
            flat_tensor_list = flat_tensor.tolist()
            flat_mask = mask_layer.flatten()
            flat_mask_list = flat_mask.tolist()
            layer = [ ckks.EncryptionManager().encrypt_vector(flat_tensor_list[i]) if flat_mask_list[i] == 1 else flat_tensor_list[i] for i in range(len(flat_mask_list))]
            
            # for i in mask_layer:
            #     flat_tensor_list[i] = ckks.EncryptionManager().encrypt_vector(flat_tensor_list[i])
            # layer = flat_tensor_list
        clients_list[k].append(layer)
        del layer
    # encryption_queue.put((k, temp))
    # return temp

