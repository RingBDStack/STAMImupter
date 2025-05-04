import torch
import numpy as np
import pywt
import math
from model.GAT import GAT_layer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from model.downstream_MoE import SP_TSFormer_MoE, SP_TSFormer_MoE_v2
import random
import torch.optim as optim
import pandas as pd
from utils import sample_mask
from einops import rearrange
from model.GWnet.model import gwnet
import scipy.sparse as sp


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if torch.mean(loss) > 300:
        print(loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

def seq2instance(data, P, Q, mask):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, nodes, dims))
    y = np.zeros(shape=(num_sample, Q, nodes, dims))
    m = np.zeros(shape=(num_sample, P, nodes, 1))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
        m[i] = mask[i: i + P]
    return x, y, m

def disentangle(x, w, j):
    x = x.transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)

    return xl, xh
def loadGraph(spatial_graph):
    # calculate spatial and temporal graph wavelets
    adj = np.load(spatial_graph, allow_pickle=True)
    adj = adj + np.eye(adj.shape[0])
    # if os.path.exists(temporal_graph):
    #     tem_adj = np.load(temporal_graph)
    # else:
    #     tem_adj = construct_tem_adj(data, adj.shape[0])
    #     np.save(temporal_graph, tem_adj)
    # spawave = get_eigv(adj, dims)
    # temwave = get_eigv(tem_adj, dims)
    # log_string(log, f'Shape of graphwave eigenvalue and eigenvector: {spawave[0].shape}, {spawave[1].shape}')

    # derive neighbors
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]

    # log_string(log, f'Shape of localadj: {localadj.shape}')
    # return localadj, spawave, temwave
    return localadj


def loadData(filepath, P, Q, train_ratio, test_ratio, h5mode=False, miss_rate=0.25, series_in_day=288, mode='point'):
    # Traffic
    if h5mode:
        Traffic = pd.read_hdf(filepath)
        Traffic = np.expand_dims(Traffic.values, axis=1)
    else:
        Traffic = np.load(filepath, allow_pickle=True)['data']
        if Traffic.ndim == 3:
            Traffic = Traffic[:, :, :1]
        else:
            Traffic = np.expand_dims(Traffic, axis=-1)
    T, N, D = Traffic.shape
    if mode == 'block':
        M_in = sample_mask((T, N, D), 0.02, 0.05, 48, 12)
        M = np.ones_like(M_in)
        M[M_in == 1] = 0
    else:
        full_arg = list(range(Traffic.shape[0] * Traffic.shape[1]))
        random.shuffle(full_arg)
        miss_arg = full_arg[:int(miss_rate * len(full_arg))]
        miss_Traffic = np.ones_like(Traffic).flatten()
        miss_Traffic[miss_arg] = 0
        M = miss_Traffic.reshape(T, N, D)
    # miss_Traffic = Traffic.flatten()
    # miss_Traffic[miss_arg] = 0
    # M = miss_Traffic.copy()
    # M[M != 0] = 1
    # M = M.reshape(T, N, D)
    # miss_Traffic = miss_Traffic.reshape(T, N, D)
    # Traffic = miss_Traffic

    num_step = Traffic.shape[0]
    TE = np.zeros([num_step, 2])
    TE[:, 1] = np.array([i % series_in_day for i in range(num_step)])
    TE[:, 0] = np.array([(i // series_in_day) % 7 for i in range(num_step)])
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)
    Data = np.concatenate((Traffic, TE_tile),axis=-1)
    # train/val/test
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    trainData = Data[: train_steps]
    trainMask = M[:train_steps]
    valData = Data[train_steps: train_steps + val_steps]
    valMask = M[train_steps: train_steps + val_steps]
    testData = Data[-test_steps:]
    testMask = M[-test_steps:]
    # testData = Data[train_steps:test_steps + train_steps]
    # testMask = M[train_steps:test_steps + train_steps]
    # valData = Data[-val_steps:]
    # valMask = M[-val_steps:]
    # X, Y
    trainX, trainY, trainM = seq2instance(trainData, P, Q, trainMask)
    valX, valY, valM = seq2instance(valData, P, Q, valMask)
    testX, testY, testM = seq2instance(testData, P, Q, testMask)

    # normalization
    mean, std = [], []
    mean.append(np.mean(trainX[:, :, :, 0]))
    std.append(np.std(trainX[:, :, :, 0]))
    mean.append(np.mean(trainX[:, :,:,  0]))
    std.append(np.std(trainX[:, :, :, 0]))
    mean.append(np.mean(trainX[:, :,:,  0]))
    std.append(np.std(trainX[:, :, :, 0]))
    # mean.append(np.mean(valX[:, :, 0]))
    # std.append(np.std(valX[:, :, 0]))
    # mean.append(np.mean(testX[:, :, 0]))
    # std.append(np.std(testX[:, :, 0]))

    return trainX, trainM, valX, valM, testX, testM, mean, std, trainData[..., 0], trainY, valY, testY


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_gwnet_adj(path, device):
    adj = np.load(path, allow_pickle=True)
    adj = adj + np.eye(adj.shape[0])
    adj = [asym_adj(adj), asym_adj(np.transpose(adj))]
    adj = [torch.tensor(i).to(device) for i in adj]
    return adj




epoch = 200
patience = 6
input_dim = 5
emb_dim = 32
Tembed_dim = 64
output_dim = 1
num_nodes = 263
num_series = 24
num_heads = 4
mlp_ratio = 4
dropout = 0.1
num_layers = 4
batch_size = 8
point_miss_rate = 0.60
device = "cuda:0"
data = 'nyc_gwnet_impute_60_dg'
adj_path = "data/NYCTAXI/adj.npy"
pre = True
full = False
pre_dg = True

trainX, trainM, valX, valM, testX, testM, mean, std, trainData, trainY, valY, testY = loadData("data/NYCTAXI/nyctx_77.npz", 24, 12, 0.7, 0.2, miss_rate=point_miss_rate, series_in_day=24, mode='point')
# trainX, trainM, valX, valM, testX, testM, mean, std, trainData = loadData("data/PEMS04/PEMS04.npz", 24, 12, 0.7, 0.1, miss_rate=point_miss_rate)
num_train = trainX.shape[0]
num_val = valX.shape[0]
num_test = testX.shape[0]

num_batch = math.ceil(num_train / batch_size)
pre_adj = loadGraph(adj_path)
adj = load_gwnet_adj(adj_path, device)

if pre:
    pretrained_model = SP_TSFormer_MoE_v2(input_dim, emb_dim, Tembed_dim, output_dim, num_nodes, num_series, pre_adj, num_heads, mlp_ratio, dropout, num_layers).to(device)
    pretrained_model.load_state_dict(torch.load("checkpoints/nyc_60_pre_best_val_mae_5.7.pth"))
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # optimizer_pre = optim.Adam(pretrained_model.dyna_layer.parameters(), lr=0.001, weight_decay=0.0001)
model = gwnet(device, num_nodes, in_dim=3, dropout=0.3, supports=adj, addaptadj=False, aptinit=None, dg_bool=pre_dg).to(device)
# model.load_state_dict(torch.load("checkpoints/nyc_gwnet_impute_25_dg_best_val_mae_8.18.pth"))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
best_val_loss = None
wait = 0

for j in range(epoch):
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainM = trainM[permutation]
    trainY = trainY[permutation]
    train_loss = []
    train_mape = []
    train_rmse = []
    for i in range(num_batch):
        model.train()
        optimizer.zero_grad()
        # if pre:
        #     optimizer_pre.zero_grad()
        si = i * batch_size
        ei = min(num_train, (i + 1) * batch_size)
        trainx = trainX[si:ei].copy()
        trainm = trainM[si:ei].copy()
        trainy = torch.from_numpy(trainY[si:ei].copy())[:, :, :, :1].float().to(device)
        trainx[:, :, :, 0] = (trainx[:, :, :, 0] - mean[0]) / std[0]
        if not full:
            trainx[:, :, :, :1] = trainx[:, :, :, :1] * trainm
        if pre:
            xl, xh = disentangle(trainx[:, :, :, :1], 'db1', 1)
            res, A, M = pretrained_model(trainx, xl, xh)
            res = torch.cat((res, torch.from_numpy(trainx[:, :, :, 1:]).float().to(device)), dim=-1)
            res = torch.where(torch.from_numpy(trainm).bool().to(device), torch.from_numpy(trainx).float().to(device), res).transpose(1, 3)
            if not pre_dg:
                res = (model(res)) * std[0] + mean[0]
            else:
                res = (model(res, A, M)) * std[0] + mean[0]
        else:
            trainx = torch.from_numpy(trainx).float().to(device).transpose(1, 3)
            res = (model(trainx)) * std[0] + mean[0]
        loss = masked_mae(res, trainy, 0.0)
        mape = masked_mape(res, trainy, 0.0)
        rmse = masked_rmse(res, trainy, 0.0)
        loss.backward()
        optimizer.step()
        # if pre:
        #     optimizer_pre.step()
        train_loss.append(loss.item())
        train_mape.append(mape.item())
        train_rmse.append(rmse.item())
        if i % 100 == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            print(log.format(i, loss, mape, rmse))

    val_loss, val_mape, val_rmse = [], [], []
    val_batch = math.ceil(valX.shape[0] / batch_size)
    for i in range(val_batch):
        model.eval()
        si = i * batch_size
        ei = min(num_val, (i + 1) * batch_size)
        valx = valX[si:ei].copy()
        valm = valM[si:ei].copy()
        valy = torch.from_numpy(valY[si:ei].copy())[:, :, :, :1].float().to(device)
        valx[:, :, :, 0] = (valx[:, :, :, 0] - mean[0]) / std[0]
        if not full:
            valx[:, :, :, :1] = valx[:, :, :, :1] * valm
        if pre:
            xl, xh = disentangle(valx[:, :, :, :1], 'db1', 1)
            res, A, M = pretrained_model(valx, xl, xh)
            res = torch.cat((res, torch.from_numpy(valx[:, :, :, 1:]).float().to(device)), dim=-1)
            res = torch.where(torch.from_numpy(valm).bool().to(device),
                              torch.from_numpy(valx).float().to(device), res).transpose(1, 3)
            if not pre_dg:
                res = (model(res)) * std[0] + mean[0]
            else:
                res = (model(res, A, M)) * std[0] + mean[0]
        else:
            valx = torch.from_numpy(valx).float().to(device).transpose(1, 3)
            res = (model(valx)) * std[0] + mean[0]
        loss = masked_mae(res, valy, 0.0)
        mape = masked_mape(res, valy, 0.0)
        rmse = masked_rmse(res, valy, 0.0)
        val_loss.append(loss.item())
        val_mape.append(mape.item())
        val_rmse.append(rmse.item())
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}.  Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}'
    print(log.format(j + 1, np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), np.mean(val_loss), np.mean(val_mape), np.mean(val_rmse)))
    if best_val_loss is None:
        best_val_loss = np.mean(val_loss)
        torch.save(model.state_dict(), "checkpoints/" + data + "_best_val_mae.pth")
    elif np.mean(val_loss) < best_val_loss:
        wait = 0
        best_val_loss = np.mean(val_loss)
        torch.save(model.state_dict(), "checkpoints/" + data + "_best_val_mae.pth")
        print("best model ")
    wait += 1
    if wait > patience:
        break

    if (j % 5 == 0) or wait == 1:
        test_batch = math.ceil(testX.shape[0] / batch_size)
        test_loss = []
        test_mape = []
        test_rmse = []
        for i in range(test_batch):
            model.eval()
            si = i * batch_size
            ei = min(num_test, (i + 1) * batch_size)
            testx = testX[si:ei].copy()
            testm = testM[si:ei].copy()
            testy = torch.from_numpy(testY[si:ei].copy())[:, :, :, :1].float().to(device)
            testx[:, :, :, 0] = (testx[:, :, :, 0] - mean[0]) / std[0]
            if not full:
                testx[:, :, :, :1] = testx[:, :, :, :1] * testm
            if pre:
                xl, xh = disentangle(testx[:, :, :, :1], 'db1', 1)
                res, A, M = pretrained_model(testx, xl, xh)
                res = torch.cat((res, torch.from_numpy(testx[:, :, :, 1:]).float().to(device)), dim=-1)
                res = torch.where(torch.from_numpy(testm).bool().to(device),
                                  torch.from_numpy(testx).float().to(device), res).transpose(1, 3)
                if not pre_dg:
                    res = (model(res)) * std[0] + mean[0]
                else:
                    res = (model(res, A, M)) * std[0] + mean[0]
            else:
                testx = torch.from_numpy(testx).float().to(device).transpose(1, 3)
                res = (model(testx)) * std[0] + mean[0]
            loss = masked_mae(res, testy, 0.0)
            mape = masked_mape(res, testy, 0.0)
            rmse = masked_rmse(res, testy, 0.0)
            test_loss.append(loss.item())
            test_mape.append(mape.item())
            test_rmse.append(rmse.item())
        log = "Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}"
        print(log.format(np.mean(test_loss), np.mean(test_mape), np.mean(test_rmse)))


model.load_state_dict(torch.load("checkpoints/" + data + "_best_val_mae.pth"))
# model.load_state_dict(torch.load(data + "_best_val_mae.pth"))
test_batch = math.ceil(testX.shape[0] / batch_size)
test_loss = []
test_mape = []
test_rmse = []
for i in range(test_batch):
    model.eval()
    si = i * batch_size
    ei = min(num_test, (i + 1) * batch_size)
    testx = testX[si:ei].copy()
    testm = testM[si:ei].copy()
    testy = torch.from_numpy(testY[si:ei].copy())[:, :, :, :1].float().to(device)
    testx[:, :, :, 0] = (testx[:, :, :, 0] - mean[0]) / std[0]
    if not full:
        testx[:, :, :, :1] = testx[:, :, :, :1] * testm
    if pre:
        xl, xh = disentangle(testx[:, :, :, :1], 'db1', 1)
        res, A, M = pretrained_model(testx, xl, xh)
        res = torch.cat((res, torch.from_numpy(testx[:, :, :, 1:]).float().to(device)), dim=-1)
        res = torch.where(torch.from_numpy(testm).bool().to(device),
                          torch.from_numpy(testx).float().to(device), res).transpose(1, 3)
        if not pre_dg:
            res = (model(res)) * std[0] + mean[0]
        else:
            res = (model(res, A, M)) * std[0] + mean[0]
    else:
        testx = torch.from_numpy(testx).float().to(device).transpose(1, 3)
        res = (model(testx)) * std[0] + mean[0]
    loss = masked_mae(res, testy, 0.0)
    mape = masked_mape(res, testy, 0.0)
    rmse = masked_rmse(res, testy, 0.0)
    test_loss.append(loss.item())
    test_mape.append(mape.item())
    test_rmse.append(rmse.item())
log = "Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}"
print(log.format(np.mean(test_loss), np.mean(test_mape), np.mean(test_rmse)))
torch.save(model.state_dict(), "checkpoints/" + data + "_best_val_mae_" + str(round(np.mean(test_loss), 2)) + ".pth")


