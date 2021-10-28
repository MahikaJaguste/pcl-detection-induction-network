import configparser
import pickle
import os
import torch
from torch import optim
import random
import numpy as np
from model import FewShotInduction
from criterion import Criterion
from tensorboardX import SummaryWriter


def train(episode):
    model.train()
    data, target = train_loader.get_batch()
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    predict = model(data)
    loss, acc = criterion(predict, target)
    loss.backward()
    optimizer.step()

    writer.add_scalar('train_loss', loss.item(), episode)
    writer.add_scalar('train_acc', acc, episode)
    if episode % log_interval == 0:
        print('Train Episode: {} Loss: {} Acc: {}'.format(episode, loss.item(), acc))
    
    return acc


def dev(episode):
    model.eval()
    correct = 0.
    count = 0.
    for data, target in dev_loader:
        data = data.to(device)
        target = target.to(device)
        predict = model(data)
        # print("predict ", predict)
        # print("target", target)
        # print("length of target", len(target))
        _, acc = criterion(predict, target)
        # print("dev acc = ",acc)
        amount = len(target) - support * 5
        # print("dev amount = ",amount)
        correct += acc * amount
        # print("correct = ", correct)
        count += amount
        # print("count = ", count)
    acc = correct / count
    writer.add_scalar('dev_acc', acc, episode)
    print('Dev Episode: {} Acc: {}'.format(episode, acc))
    return acc


def test():
    model.eval()
    correct = 0.
    count = 0.
    y_pred = []
    y_test = []

    way=int(config['model']['class'])
    shot=int(config['model']['support'])
    print(way)
    print(shot)
    amount = way*shot
    print(amount)

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        predict = model(data)
        ####################################

        print(type(target[amount:].tolist()), target[amount:].tolist())
        print(type(torch.argmax(predict, dim=1).tolist()),torch.argmax(predict, dim=1).tolist())

        y_test.append(target[amount:].tolist())
        y_pred.append(torch.argmax(predict, dim=1).tolist())

        ###################################
        _, acc = criterion(predict, target)
        amount = len(target) - support * 5
        correct += acc * amount
        count += amount
    acc = correct / count
    writer.add_scalar('test_acc', acc)
    print('Test Acc: {}'.format(acc))
    return acc, y_test, y_pred


def main():
    # best_episode, best_acc = 0, 0.
    # best_train = 0.0
    # best_train_ep = 0
    # episodes = int(config['model']['episodes'])
    # early_stop = int(config['model']['early_stop']) * dev_interval
    # for episode in range(1, episodes + 1):

    #     train_acc = train(episode)
    #     if(train_acc > best_train):
    #         best_train = train_acc
    #         best_train_ep = episode

    #     if episode % dev_interval == 0:
    #         acc = dev(episode)
    #         if acc > best_acc:
    #             print('Better acc! Saving model!')
    #             torch.save(model.state_dict(), config['model']['model_path'])
    #             best_episode, best_acc = episode, acc
    #         if episode - best_episode >= early_stop:
    #             print('Early stop at episode', episode)
    #             break

    # print('Best training accuracy is ', best_train, 'on episode', best_train_ep)
    # print('Reload the best model on episode', best_episode, 'with best acc', best_acc.item())
    # ckpt = torch.load(config['model']['model_path'])
    # model.load_state_dict(ckpt)
    test()


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['model']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # log_interval
    log_interval = int(config['model']['log_interval'])
    dev_interval = int(config['model']['dev_interval'])

    # data loaders
    train_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['train_loader']), 'rb'))
    dev_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'rb'))
    test_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['test_loader']), 'rb'))

    vocabulary = pickle.load(open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'rb'))

    # word2vec weights
    weights = pickle.load(open(os.path.join(config['data']['path'], config['data']['weights']), 'rb'))

    # model & optimizer & criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support = int(config['model']['support'])
    model = FewShotInduction(C=int(config['model']['class']),
                             S=support,
                             vocab_size=len(vocabulary),
                             embed_size=int(config['model']['embed_dim']),
                             hidden_size=int(config['model']['hidden_dim']),
                             d_a=int(config['model']['d_a']),
                             iterations=int(config['model']['iterations']),
                             outsize=int(config['model']['relation_dim']),
                             weights=weights).to(device)

    '''
    C = 5 = number of classes
    S = 5 = number of samples in support set for each class
    vocab_size = size of vocabulary built from training set using Vocabulary()
    embed_size = 300 = length of word2vec vectors
    hidden_size = 128
    d_a = 64 = hyperparamter used in Eq 3 = the length of the output vector of the encoder module
    iterations = 3
    outsize = 100 (relation_dim)
    weights = word2ve weights obtained on training data
    '''
    model.load_state_dict(torch.load(config['model']['model_path']))
    optimizer = optim.Adam(model.parameters(), lr=float(config['model']['lr']))
    criterion = Criterion(way=int(config['model']['class']),
                          shot=int(config['model']['support']))

    # writer
    os.makedirs(config['model']['log_path'], exist_ok=True)
    writer = SummaryWriter(config['model']['log_path'])
    main()
    writer.close()
