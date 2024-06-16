import model
from client_multi import *
# from client import *
from utils.aggregator import *
from utils.dispatchor import *
from utils.optimizer import *
# from utils.clusteror import *
from utils.global_test import *
from utils.local_test import *
from utils.sampling import *
from utils.AnchorLoss import *
from utils.ContrastiveLoss import *
from utils.CKA import linear_CKA, kernel_CKA
import time


def seed_torch(seed, test = True):
    if test:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Server:

    def __init__(self, args, model, dataset, synthetic_dataset, dict_users, synthetic_dict_users):
        seed_torch(args.seed)
        self.args = args
        self.nn = copy.deepcopy(model)
        self.nns = [[] for i in range(self.args.K)]
        self.p_nns = []
        self.cls = []
        self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict =  dict((k, [0]) for k in key)
        #self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict =  dict((i, []) for i in range(args.r))
        self.dataset = dataset
        self.dict_users = dict_users
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_dict_users = synthetic_dict_users


        # store global model in the list
        self.global_list = []
        for layer in self.nn.parameters():
            self.global_list.append(layer)
        # store all clients model in the list
        self.clients_list = [[] for i in range(self.args.K)]
        # store shape list of the model
        self.shape_list= []
        for layer in model.parameters():
            self.shape_list.append(layer.shape)

        
        

        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2) 
            
        self.contrastiveloss = ContrastiveLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.contrastiveloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.contrals.append(temp2) 

    def fedfa_anchorloss(self, testset, dict_users_test, mask, similarity=False, fedbn = False,
                         test_global_model_accuracy = False):
        acc_list = []
        similarity_dict = {"feature":[], "classifier":[]}
        if fedbn:
            acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
            datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
        else:
            acc_list = []
        
        for t in range(self.args.r):
            print('round', t + 1, ':')
            # isAuthentic = True if t%2 == 0 else False
            isAuthentic = True
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C is client sample rate
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # sample m clients
            self.index_dict[t]= index

            # dispatch
            if fedbn:
                for i in index:
                    global_w = self.nn.state_dict()
                    client_w = self.nns[i].state_dict()
                    for key in global_w:
                        if 'bn' not in key:
                            client_w[key] = global_w[key] 
                    self.nns[i].load_state_dict(client_w)
            else:
                # dispatch(index, self.nn, self.nns)
                safe_dispatch(index, self.global_list, self.clients_list)
            dispatch(index, self.anchorloss, self.cls)

            if isAuthentic:
                print("Authentic!")
                dataset_selected = self.dataset
                dict_users_selected = self.dict_users
            else:
                dataset_selected = self.synthetic_dataset
                dict_users_selected = self.synthetic_dict_users

            #joint updating to obtain personalzied model based on updating global model
            # start_time = time.time()
            self.cls, self.nns, self.loss_dict, self.clients_list  = client_fedfa_cl(self.args,index, self.cls, self.nns, self.nn, t, dataset_selected, dict_users_selected, self.loss_dict, mask, self.global_list, self.clients_list, self.shape_list, isAuthentic) 
            # end_time = time.time()
            # print("Execution time for this round: ", end_time - start_time)
            
            # aggregation
            if fedbn:
                aggregation(index, self.nn, self.nns, self.dict_users,fedbn=True)
            else:
                if isAuthentic:
                    safe_aggregation(index, self.dict_users, self.global_list, self.clients_list)
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users)
            aggregation(index, self.anchorloss, self.cls, self.dict_users)


            if test_global_model_accuracy:
                if fedbn:
                    for index1, testset_per in enumerate(testset):
                        acc,_ = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index1]])
                        acc_list_dict[datasets_name[index1]].append(acc)
                        print(acc)
            
                else:
                    if not isAuthentic:
                        temp = copy.deepcopy(self.nn.state_dict())
                        for key,layer in zip(temp, self.global_list):
                            temp[key] = layer
                        self.nn.load_state_dict(temp)

                    acc,_ = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                    print(acc)
            
        if fedbn:
            mean_CKA_dict = acc_list_dict
        else:
            mean_CKA_dict = acc_list

        for k in range(self.args.K):
            path="results/Test/{} skew/{}/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(self.args.skew,
                                                self.args.dataset, self.args.seed,k, self.args.E, self.args.split)
            if self.nns[k]!=[]:
                torch.save(self.nns[k].state_dict(), path)
        self.nns = [[] for i in range(self.args.K)]
        self.clients_list = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, mean_CKA_dict


def compute_mean_feature_similarity(args, index, client_models, trainset, dict_users_train, testset, dict_users_test):
        pdist = nn.PairwiseDistance(p=2)
        dict_class_verify = {i: [] for i in range(args.num_classes)}
        for i in dict_users_test:
            for c in range(args.num_classes):
                if np.array(testset.targets)[i] == c:
                    dict_class_verify[c].append(i)
        #dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}
        dict_clients_features = {k: [] for k in index}
        for k in index:
            # labels = np.array(trainset.targets)[list(dict_users_train[k])]
            # labels_class = set(labels.tolist())
            #for c in labels_class:
            for c in range(args.num_classes):
                features_oneclass = verify_feature_consistency(args, client_models[k], testset,
                                                                        dict_class_verify[c])
                features_oneclass = features_oneclass.view(1,features_oneclass.size()[0],
                                                            features_oneclass.size()[1])
                if c ==0:
                    dict_clients_features[k] = features_oneclass
                else:
                    dict_clients_features[k] = torch.cat([dict_clients_features[k],features_oneclass])
                
        cos_sim_matrix = torch.zeros(len(index),len(index))
        for p, k in enumerate(index):
            for q, j in enumerate(index):
                for c in range(args.num_classes):
                    cos_sim0 = pdist(dict_clients_features[k][c],
                                    dict_clients_features[j][c])
                    # cos_sim0 = torch.cosine_similarity(dict_clients_features[k][c],
                    #                                   dict_clients_features[j][c])
                    # cos_sim0 = get_cos_similarity_postive_pairs(dict_clients_features[k][c],
                    #                                    dict_clients_features[j][c])
                    if c ==0:
                        cos_sim = cos_sim0
                    else:
                        cos_sim = torch.cat([cos_sim,cos_sim0])
                cos_sim_matrix[p][q] = torch.mean(cos_sim)
        mean_feature_similarity = torch.mean(cos_sim_matrix)

        return mean_feature_similarity
