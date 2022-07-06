import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# from utils import cal_std_mean_and_get_plot_data, save_std_mean, plot_all_model

def cal_std_mean_and_get_plot_data(my_data, model_name_list):
    final_list = []
    plot_data = []
    for data in my_data:
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        
        p_train_acc = []
        p_train_loss = []
        p_test_acc = []
        p_test_loss = []
        t_count = 0
        for info in data:
            if t_count == 0:
                p_train_acc.extend(info[0])
                p_train_loss.extend(info[1])
                p_test_acc.extend(info[4])
                p_test_loss.extend(info[5])
            train_acc.append(info[0][info[-1]])
            train_loss.append(info[1][info[-1]])
            test_acc.append(info[4][info[-1]])
            test_loss.append(info[5][info[-1]])
            t_count += 1
        final_list.append([train_acc, train_loss, test_acc, test_loss])
        plot_data.append([p_train_acc, p_train_loss, p_test_acc, p_test_loss])
    print("\nUse Best Test Acc to See Result\n")
    for model, m_name in zip(final_list, model_name_list):
        print("Model: {}".format(m_name))
        print("Train acc: {:.2f}\u00B1{:.2f}".format(np.mean(model[0])*100, np.std(model[0])*100))
        print("Test acc: {:.2f}\u00B1{:.2f}".format(np.mean(model[2])*100, np.std(model[2])*100))
        print("Train loss: {:.3f}\u00B1{:.3f}".format(np.mean(model[1]), np.std(model[1])))
        print("Test loss: {:.3f}\u00B1{:.3f}\n".format(np.mean(model[3]), np.std(model[3])))  
    final_list2 = []
    plot_data2 = []
    for data in my_data:
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
        
        p_train_acc = []
        p_train_loss = []
        p_test_acc = []
        p_test_loss = []
        t_count = 0
        for info in data:
            if t_count == 0:
                p_train_acc.extend(info[0])
                p_train_loss.extend(info[1])
                p_test_acc.extend(info[4])
                p_test_loss.extend(info[5])
            train_acc.append(info[0][info[-3]])
            train_loss.append(info[1][info[-3]])
            test_acc.append(info[4][info[-3]])
            test_loss.append(info[5][info[-3]])
            t_count += 1
        final_list2.append([train_acc, train_loss, test_acc, test_loss])
        plot_data2.append([p_train_acc, p_train_loss, p_test_acc, p_test_loss])
    print("\nUse Best Valid Acc to See Result\n")
    for model, m_name in zip(final_list2, model_name_list):
        print("Model: {}".format(m_name))
        print("Train acc: {:.2f}\u00B1{:.2f}".format(np.mean(model[0])*100, np.std(model[0])*100))
        print("Test acc: {:.2f}\u00B1{:.2f}".format(np.mean(model[2])*100, np.std(model[2])*100))
        print("Train loss: {:.3f}\u00B1{:.3f}".format(np.mean(model[1]), np.std(model[1])))
        print("Test loss: {:.3f}\u00B1{:.3f}\n".format(np.mean(model[3]), np.std(model[3])))  
    return final_list, plot_data, final_list2, plot_data2

def save_std_mean(final_result, save_loc):
    print("\nThere have four type data\n")
    final_result = np.array(final_result)
    np.save(save_loc, final_result)
    print("\nFinish save to npy\n")
    
def plot_all_model(save_loc, final_result, model_name, figsize=(8,8), title="combine all models"):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    for i, name in zip(final_result, model_name):
        ax1.plot(range(1, len(i[0])+1), i[0], label=name)
        ax1.set_title("Train Acc")
    #     ax1.yaxis.set_ticks(np.arange(0.23, 0.995, 0.05))
        ax1.legend()

        ax2.plot(range(1, len(i[1])+1), i[1], label=name)
    #     ax3.yaxis.set_ticks(np.arange(0, 3, 0.1))
        ax2.set_title("Train Loss")
        ax2.legend()

        ax3.plot(range(1, len(i[2])+1), i[2], label=name)
        ax3.set_title("Test Acc")
    #     ax2.yaxis.set_ticks(np.arange(0.25, 0.96, 0.03))
        ax3.legend()

        ax4.plot(range(1, len(i[3])+1), i[3], label=name)
    #     ax4.yaxis.set_ticks(np.arange(0, 3, 0.1))
        ax4.set_title("Test Loss")
        ax4.legend()
    plt.savefig(save_loc)
def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """
    store = []
    for i, _ in tqdm(cifar100_dataset):
        store.append(np.array(i))
    store = np.array(store)
    print(store.shape)
    data_r = np.dstack([store[i][:, :] for i in range(len(cifar100_dataset))]) / 255
#     data_g = np.dstack([store[i][:, :, 1] for i in range(len(cifar100_dataset))]) / 255
    mean = np.mean(data_r)
    std = np.std(data_r)

    return mean, std

def get_result_table(Result):
    print("Show Final Result with Best Valid Accuracy\n\n")
    print("{:<18}|{:<18}|{:<18}".format("Model Name", "Train Acc", "Test Acc"))
    for i in Result:
        print("{:<18}|{:<18.4f}|{:<18.4f}".format(i.model_name, i.train_acc[i.best_acc_idx], i.test_acc[i.best_acc_idx]))
