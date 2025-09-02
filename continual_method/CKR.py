import copy
import itertools
import random

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from model.network import Network
from matric.metric import valid
from torch.utils.data import Dataset
import argparse
from loss import Loss

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from replay_dataset import replay_dataset
from splitDataset import *
from util import qs_cls_inc, qs_cls_inc_single

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from profiling_utils import CLProfiler
prof = CLProfiler()
# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
dataname = 'Fashion'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=100)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=256)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# lamdas = [1,0,0.8,0.6,0.4,0.2]
# temperatures = [0.1,0.3,0.5,0.7,0.9]

lamdas = [1]
temperatures = [0.5]

# device = 'cpu'
# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 60
    seed = 50
if args.dataset == "BDGP":
    args.con_epochs = 20
    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 50
    seed = 3
if args.dataset == "Fashion":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 50
    seed = 5
else:
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)
# class_list = [[1,8],[4,9],[3,7],[5,0],[2,6]]
# class_list = [[1,2,3,4,5],[0,6,7,8,9]]
# task_range = [5,10]
# task_num = 2
# 写一个方法，能够随机生成list，其中包含两个长度为5的list，整个list从0-9
# def generate_random_lists():
#     # 创建包含0到9的列表
#     numbers = list(range(10))
#     # 随机打乱列表
#     random.shuffle(numbers)
#
#     # 分割列表为5个子列表，每个子列表长度为2
#     lists = [numbers[i:i + 2] for i in range(0, len(numbers), 2)]
#     return lists
# class_list = generate_random_lists()
# print(class_list)

class_list = [[1,9],[3,7],[2,8],[4,6],[5,0]]
task_range = [2,4,6,8,10]
task_num = 5

# class_list = [[0,1,2,3,4]]
# task_range = [5]
# task_num = 1
#
# class_list = [[1, 3], [0, 2, 4]]
# task_range = [2, 5]
# task_num = 2

# class_list = [[0,2],[3,4],[1,5,6]]
# task_range = [2,4,7]
# task_num = 3

# class_list = [[0,1,2,3,4,5,6,7,8,9]]
# task_range = [10]
# task_num = 1

# dataset, dims, view, data_size, class_num = load_data('Fashion')
dataset, dims, view, data_size, class_num = load_data(args.dataset)
# view = 1
data = load_data_split(dataset, class_list,view)
# class_num = 2
data_size = 800
# 将data分为训练集和测试集
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False
)
train_datasets = []
test_datasets = []


for i in range(task_num):
    split_dataset = getSplitData(dataname,data[i])
    train_size = int(0.8 * len(split_dataset))
    test_size = len(split_dataset) - train_size
    train_dataset, test_dataset = random_split(split_dataset, [train_size, test_size])
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)



train_data_loaders = {}
test_data_loaders = {}
for i in range(task_num):
    train_data_loader = torch.utils.data.DataLoader(
            train_datasets[i],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )
    test_data_loader = torch.utils.data.DataLoader(
            test_datasets[i],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )
    train_data_loaders[i] = train_data_loader
    test_data_loaders[i] = test_data_loader
#
# dataset, dims, view, data_size, class_num = load_data(args.dataset)
#
# data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True,
#     )


def pretrain(epoch,i):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(train_data_loaders[i]):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs,i)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    loss_values.append(tot_loss)
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(train_data_loaders[i])))



def list_to_dataset(replay_buffer):

    # 将list格式的replay buffer转化为 dataset格式
    dataset = replay_dataset(replay_buffer,view)
    return dataset


def merge_dataset():
    task_data = []
    loader = torch.utils.data.DataLoader(
        train_datasets[i],
        batch_size=len(train_datasets[i]),
        shuffle=False,
    )

    model.eval()

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, xrs, _ = model.forward(xs,i)
        for v in range(view):
            xs[v] = xs[v].cpu()
        task_data.append(xs)
    return task_data





def pretrain_with_replay(epoch,i):

    # 将list格式的replay buffer转化为 dataset格式
    datasets = []
    for task_num in range(i):
        dataset = list_to_dataset(replay_buffer[task_num])
        datasets.append(dataset)
    # 把多个dataset merge 成一个
    task_list = merge_dataset()
    task_dataset = list_to_dataset(task_list[0])

    datasets.append(task_dataset)
    dataset_merge = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset_merge,
        batch_size=len(dataset_merge),
        shuffle=True,
    )

    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs, i)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(loader)))





def contrastive_train_with_replay(epoch,i):
    # loader = torch.utils.data.DataLoader(
    #     train_datasets[i],
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=True
    # )
    datasets = []
    for task_num in range(i):
        dataset = list_to_dataset(replay_buffer[task_num])
        datasets.append(dataset)
    # 把多个dataset merge 成一个
    task_list = merge_dataset()
    task_dataset = list_to_dataset(task_list[0])

    datasets.append(task_dataset)
    dataset_merge = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset_merge,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # loader = torch.utils.data.DataLoader(
    #     train_datasets[i],
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=True
    # )


    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs,i)

        qs = qs_cls_inc(qs, view, class_list, i, device)
        loss_list = []

        if i != 0:
            phs, pqs, pxrs, pzs = pre_model(xs, i)
            pqs = qs_cls_inc(pqs, view, class_list, i, device)
            for v in range(view):
                loss_list.append(criterion.forward_feature(hs[v], phs[v]) * temperature)
                loss_list.append(criterion.forward_label(qs[v], pqs[v]) * temperature)

                # loss_list.append(criterion.forward_feature_RINCE(qs[v], pqs[v]) * temperature)




        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]) * (1 - temperature))
                loss_list.append(criterion.forward_label(qs[v], qs[w]) * (1 - temperature))
            # loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    loss_values.append(tot_loss)
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(train_data_loaders[i])))
def contrastive_train(epoch,i):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, y, _) in enumerate(train_data_loaders[i]):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs,i)

        qs = qs_cls_inc(qs, view, class_list, i, device)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            # loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    loss_values.append(tot_loss)
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(train_data_loaders[i])))

def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        train_datasets[i],
        batch_size=len(train_datasets[i]),
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs,i)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(len(train_datasets[i]), 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def make_replay_list(model, device, lamda):
    if lamda == 0:
        return []
    loader = torch.utils.data.DataLoader(
        train_datasets[i],
        batch_size= int(len(train_datasets[i]) * lamda),
        shuffle=True,
    )

    model.eval()

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, xrs, _ = model.forward(xs,i)
        for v in range(view):
            xrs[v] = xrs[v].cpu()
        buffer = xrs
        break



    return buffer


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def fine_tuning(epoch, new_pseudo_label,task):

    loader = torch.utils.data.DataLoader(
        train_datasets[i],
        batch_size=len(train_datasets[i]),
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _ = model(xs,i)
        qs = qs_cls_inc(qs, view, class_list, i, device)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))


        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    loss_values.append(tot_loss)
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def fine_tuning_with_replay(epoch, new_pseudo_label, i):
    datasets = []
    for task_num in range(i):
        dataset = list_to_dataset(replay_buffer[task_num])
        datasets.append(dataset)
    # 把多个dataset merge 成一个
    task_list = merge_dataset()
    task_dataset = list_to_dataset(task_list[0])

    datasets.append(task_dataset)
    dataset_merge = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset_merge,
        batch_size=len(dataset_merge),
        shuffle=False,
        drop_last=True
    )

    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, qs, _, _ = model(xs,i)
        # qs = qs_cls_inc(qs, view, class_list, i, device, len(train_datasets[i]))
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))


        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(dataset_merge)))




def make_pseudo_label_with_replay(model, device):
    datasets = []
    for task_num in range(i):
        dataset = list_to_dataset(replay_buffer[task_num])
        datasets.append(dataset)
    # 把多个dataset merge 成一个
    task_list = merge_dataset()
    task_dataset = list_to_dataset(task_list[0])

    datasets.append(task_dataset)
    dataset_merge = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset_merge,
        batch_size=len(dataset_merge),
        shuffle=False,
        drop_last=True
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs, i)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    # kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans = KMeans(n_clusters= (i+1)*class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(len(dataset_merge), 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label

def save_loss(epochs,loss_values):
    import pandas as pd

    # 示例数据：每个epoch的loss值


    # 创建DataFrame
    data = {'Epoch': epochs, 'Loss': loss_values}
    df = pd.DataFrame(data)

    # 保存到CSV文件
    csv_file = 'loss_values_{}.csv'.format(i)
    df.to_csv(csv_file, index=False)

    print(f"Loss values have been saved to {csv_file}")


def plot_xrs():
    loader = torch.utils.data.DataLoader(
        train_datasets[i],
        batch_size=int(len(train_datasets[i]) * lamda),
        shuffle=True,
    )

    model.eval()

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, xrs, _ = model.forward(xs, i)
        for v in range(view):
            xrs[v] = xrs[v].cpu()
            xs[v] = xs[v].cpu()
        break

    # xrs为28 * 28 的图像
    for v in range(view):
        xrs[v] = xrs[v].reshape(xrs[v].shape[0], 28, 28)
        xs[v] = xs[v].reshape(xs[v].shape[0], 28, 28)

    # 画出每个视图的前25个xrs，同时画出当前的xs进行比较
    for v in range(view):
        for j in range(25):
            plt.subplot(5, 5, j + 1)
            plt.imshow(xrs[v][j], cmap='gray')
            plt.axis('off')

        plt.show()


# for lamda, temperature in itertools.product(lamdas,temperatures):
ts = [1]
for t in ts:
    lamda = 1
    temperature = 0.5
    args.temperature_l = t
    print('lamda:{}'.format(lamda),'temperature:{}'.format(temperature))
    accs = {}
    nmis = {}
    aris = {}
    purs = {}
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    labels = []

    buffer = []
    replay_buffer = []

    loss_values = []
    # ephs = 1

    for i in range(task_num):
        prof.start_task(i)
        class_num = task_range[i]
        # if i != 0:
        #     model = Network(view, dims, args.feature_dim, args.high_feature_dim, 10, device)
        #     model = model.to(device)
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        accs[i] = []
        nmis[i] = []
        aris[i] = []
        purs[i] = []
        print("ROUND:{}".format(i+1))
        setup_seed(seed)

        criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

        epoch = 1
        while epoch <= args.mse_epochs:
            if i == 0:
                pretrain(epoch,i)
                epoch += 1
                # ephs += 1
            else:
                pretrain(epoch, i)
                # pretrain_with_replay(epoch,i)
                epoch += 1
                # ephs += 1
        while epoch <= args.mse_epochs + args.con_epochs:
            if i == 0:
                with prof.measure(i, "contrastive"):
                    contrastive_train(epoch,i)
                epoch += 1
                # ephs += 1
            else:
                with prof.measure(i, "contrastive_with_replay"):
                    contrastive_train_with_replay(epoch,i)
                epoch += 1
                # ephs += 1

            if epoch == args.mse_epochs + args.con_epochs:
                for j in range(i + 1):
                    acc, nmi, ari, pur = valid(model, device, test_datasets[j], view, len(test_datasets[j]), class_num, j ,class_list, eval_h=False)


        if i == 0:
            new_pseudo_label = make_pseudo_label(model, device)
        else:
            new_pseudo_label = make_pseudo_label_with_replay(model, device)

        if i == 0:
            while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
                    fine_tuning(epoch, new_pseudo_label,i)
                    epoch += 1


        for j in range(i + 1):
            acc, nmi, ari, pur = valid(model, device, test_datasets[j], view, len(test_datasets[j]), class_num, i ,class_list, eval_h=False)

            accs[i].append(acc)
            nmis[i].append(nmi)
            aris[i].append(ari)
            purs[i].append(pur)
        replay_buffer.append(make_replay_list(model, device,lamda))
        pre_model = copy.deepcopy(model)
        prof.end_task(i)
        # plot_xrs()
    # save_loss(epochs=list(range(1, ephs)),loss_values=loss_values)
    print(class_list)
    print('dataset:', args.dataset)
    prof.dump()
    for i in range(task_num):
        # 输出每个task的acc,nmi,pur和平均值
        print('Task {}:'.format(i + 1))
        print('ACC:{}'.format(accs[i]), 'mean:{:.4f}'.format(np.mean(accs[i])))
        print('NMI:{}'.format(nmis[i]), 'mean:{:.4f}'.format(np.mean(nmis[i])))
        print('ARI:{}'.format(aris[i]), 'mean:{:.4f}'.format(np.mean(aris[i])))
        print('PUR:{}'.format(purs[i]), 'mean:{:.4f}'.format(np.mean(purs[i])))