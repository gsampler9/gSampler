import numpy as np
import argparse
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gs.utils import SeedGenerator
from tqdm import tqdm


# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df['userID'][idx])
        item_id = np.array(self.df['itemID'][idx])
        label = np.array(self.df['label'][idx], dtype=np.float32)
        return user_id, item_id, label


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # build dataset and knowledge graph
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
    userID = torch.tensor(df_dataset['userID']).to(device)
    itemID = torch.tensor(df_dataset['itemID']).to(device)
    labels = torch.tensor(df_dataset['label'], dtype=torch.float32).to(device)
    # x_test.reset_index(inplace=True, drop=True)
    # train_dataset = KGCNDataset(x_train)
    # test_dataset = KGCNDataset(x_test)
    train_seedloader = SeedGenerator(
        torch.tensor(x_train.index, dtype=torch.int64, device=device), batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_seedloader = SeedGenerator(
        torch.tensor(x_test.index, dtype=torch.int64, device=device), batch_size=args.batch_size, shuffle=True, drop_last=False)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.batch_size)

    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.l2_weight)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        for i, seeds in enumerate(tqdm(train_seedloader)):
            user_ids, item_ids, batch_labels = userID[seeds], itemID[seeds], labels[seeds]
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # print train loss per every epoch
        loss_list.append(running_loss / len(train_seedloader))

        # evaluate per every epoch
        test_loss = 0
        total_roc = 0
        with torch.no_grad():
            for i, seeds in enumerate(tqdm(test_seedloader)):
                user_ids, item_ids, batch_labels = userID[seeds], itemID[seeds], labels[seeds]
                outputs = net(user_ids, item_ids)
                test_loss += criterion(outputs, batch_labels).item()
                total_roc += roc_auc_score(batch_labels.cpu().detach().numpy(),
                                           outputs.cpu().detach().numpy())
        test_loss_list.append(test_loss / len(test_seedloader))
        auc_score_list.append(total_roc / len(test_seedloader))

        print('[Epoch {}]train_loss: '.format(epoch+1),
              running_loss / len(train_seedloader), 'test_loss: ', test_loss / len(test_seedloader), 'auc: ', total_roc / len(test_seedloader))


if __name__ == '__main__':
    # prepare arguments (hyperparameters)
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str,
                        default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int,
                        default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16,
                        help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=2,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--l2_weight', type=float,
                        default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='size of training dataset')

    args = parser.parse_args()
    if args.dataset == 'movie':
        args.n_epochs = 3
        args.neighbor_sample_size = 4
        args.dim = 32
        args.n_iter = 2
        args.batch_size = 65535
        args.l2_weight = 1e-7
        args.lr = 2e-2
    print(args)

    train(args)
