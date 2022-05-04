import time

import lightgbm as lgb
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, confusion_matrix, \
    precision_score, precision_recall_curve, f1_score, roc_curve
import os
import argparse
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.DEBUG)

TRAINSET_PATH = "/mnt/storage-data/liweijie/work/vis_sim/dataset/15ra"
VALIDATIONSET_PATH = "/mnt/storage-data/liweijie/work/vis_sim/dataset/45ra"

# 存储数据的根目录
RESULT_PATH = "./lgb_result"


def get_dataset(dataset_path):
    # 读取data
    print("reading data :", dataset_path)
    data_all = pd.read_csv(dataset_path)
    data = data_all[['abs', 'angle']]
    label = data_all[['label']]
    # print(label.info())

    return data, label


class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter("[%(asctime)s] %(message)s")  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)
        # 将下面的代码加入main即可实现对应输出
        # log.logger.debug('debug')
        # log.logger.info('info')
        # log.logger.warning('警告'),
        # log.logger.error('报错')
        # log.logger.critical('严重')
        # Logger('error.log', level='error').logger.error('error')


def predict_data(data, label, model):
    start = time.time()
    predict = model.predict(data)
    end = time.time()
    run_time = end - start
    log.logger.info("run time: {}".format(run_time))

    predict_th, accuracy, conf_m, precision, recall, f1, auc = testing_model_performance(predict, label, log)

    return predict, predict_th, accuracy, conf_m, precision, recall, f1, auc, run_time


def testing_model_performance(predict, label, log):
    predict_th = np.where(predict >= 0.5, 1, 0)

    accuracy = accuracy_score(label, predict_th)
    log.logger.info('accuarcy:{}'.format(accuracy))
    precision = precision_score(label, predict_th, pos_label=1)
    log.logger.info("precision: {}".format(precision))
    recall = recall_score(label, predict_th, pos_label=1)
    log.logger.info("recall: {}".format(recall))
    f1 = f1_score(label, predict_th, pos_label=1)
    log.logger.info("f1: {}".format(f1))
    auc = roc_auc_score(label, predict)
    log.logger.info("auc: {}".format(auc))
    conf_m = confusion_matrix(label, predict_th)
    log.logger.info('confusion_matrix: {}'.format(conf_m))

    return predict_th, accuracy, conf_m, precision, recall, f1, auc


if __name__ == "__main__":
    # 获取参数
    parser = argparse.ArgumentParser(description='model option')
    parser.add_argument('-m', '--model', type=str, default='new', help='default model or new model')
    parser.add_argument('-d', '--dataset_name', type=str, default='DNR001', help='dataset')
    args = parser.parse_args()
    lgb_model_name = args.model
    dataset_name = args.dataset_name

    # 查看文件夹是否存在，不在则创建文件夹
    result_path = os.path.join(RESULT_PATH, lgb_model_name, dataset_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print("create :", result_path)
    # 训练日志文件
    log_dir = os.path.join(result_path, "{}_{}.log".format(lgb_model_name, dataset_name))
    print("Logger dir:", log_dir)
    log = Logger(log_dir)

    params = {
        'boosting_type': 'gbdt',  # 用于指定弱学习器的类型，默认值为 ‘gbdt’，表示使用基于树的模型进行计算。
        'objective': 'binary',  # 用于指定学习任务及相应的学习目标.“binary”，二分类
        'metric': 'binary_logloss',  # 用于指定评估指标，可以传递各种评估方法组成的list。
        'verbose': -1,
        'n_jobs': -1,  # 并行运行的多线程数
        'random_state': 2021,  # 指定随机数种子。
    }
    if lgb_model_name == "default":
        params = params
    elif lgb_model_name == "new":
        params.update({
            'learning_rate': 0.1,
            'max_depth': 4,
            'num_leaves': 12,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_alpha': 162,
            'reg_lambda': 719,
            'min_child_sample': 282,
            'is_unbalance': True  # 当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
        })


    # 获取训练集
    dataset_path = os.path.join(TRAINSET_PATH, "dataset_{}.csv".format(dataset_name))
    log.logger.info("Train set:{}".format(dataset_path))
    data, label = get_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2021, stratify=label)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # lgb模型训练
    print("new lgb train :\n")
    evals_result = {}  # 记录训练结果所用
    early_stopping_rounds = 100
    num_boost_round = 1000
    start = time.time()
    n_gbm = lgb.train(params, train_data, valid_sets=[test_data], evals_result=evals_result,
                      early_stopping_rounds=early_stopping_rounds, num_boost_round=num_boost_round)
    end = time.time()
    run_time = end - start
    log.logger.info("total train time: {}".format(run_time))
    # 预测训练集数据，得到训练集的预测结果
    log.logger.info("predict train set:{}".format(dataset_name))
    predict = predict_data(data=data, label=label, model=n_gbm)

    # 保存loss
    TP_dir = os.path.join(result_path, "loss_{}.npy".format(dataset_name))
    log.logger.info("Saving training process: {}".format(TP_dir))
    np.save(TP_dir, evals_result['valid_0']['binary_logloss'])

    # 验证
    file_name_list = ["DNR001_NR01", "DNR001_NR05", "DNR001_NR10",
                      "DNR001_NR004_BR005", "DNR001_NR054_BR047"]


    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    auc_sum = 0
    run_time_sum = 0
    for name in file_name_list:
        log.logger.info("\npredict val set:{}".format(name))
        val_path = os.path.join(VALIDATIONSET_PATH, "dataset_{}.csv".format(name))
        log.logger.info("reading:{}".format(val_path))
        val_data, val_label = get_dataset(val_path)
        val_predict, val_predict_th, accuracy, conf_m, precision, recall, f1, auc, run_time = predict_data(data=val_data, label=val_label, model=n_gbm)
        accuracy_sum += accuracy
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        auc_sum += auc
        run_time_sum += run_time

        pre_result = val_predict_th.reshape(300, 1000)
        pre_reault_save_path = os.path.join(result_path, "result_{}.npy".format(name))
        # log.logger.info("result save path: {}".format(pre_reault_save_path))
        np.save(pre_reault_save_path, pre_result)

    ava_accuracy = accuracy_sum / len(file_name_list)
    ava_precision = precision_sum / len(file_name_list)
    ava_recall = recall_sum / len(file_name_list)
    ava_f1 = f1_sum / len(file_name_list)
    ava_auc = auc_sum / len(file_name_list)
    ava_run_time = run_time_sum / len(file_name_list)
    log.logger.info("\n-----ava----- \naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}\nauc:{}\n"
                    "run time:{}".format(ava_accuracy, ava_precision, ava_recall, ava_f1, ava_auc, ava_run_time))

    if dataset_name == "DNR001":
        text_mode = "w"
    else:
        text_mode = "a"
    with open("./lgb_result/{}/ava_result.txt".format(lgb_model_name), text_mode) as f:
        f.write("\n-----{}----- \naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}\nauc:{}\n"
                "run time:{}".format(dataset_name, ava_accuracy, ava_precision, ava_recall, ava_f1, ava_auc, ava_run_time))

    print("Done.")

    # roc/pr曲线数据
    # roc_curve_dir = os.path.join('./result/muser', "{}_{}_roc_curve.npy".format(lgb_model_name, train_data_model))
    # # precision, recall, thresholds = precision_recall_curve(label, predict)
    # fpr, tpr, thresholds = roc_curve(label, predict)
    # data_csv = np.dstack((fpr, tpr)).reshape(-1, 2)
    # data_csv = pd.DataFrame(data_csv, columns=['fpr', 'tpr'])
    # print('\nSave to {}'.format(roc_curve_dir))
    # data_csv.to_csv(roc_curve_dir, index=False)
    # print(data_csv.info())


    # loss图
    # TP_dir = os.path.join('./result/muser', "{}_{}_loss.npy".format(lgb_model_name, train_data_model))
    # print("Saving training process: ", TP_dir)
    # np.save(TP_dir, evals_result['valid_0']['binary_logloss'])
    #
    # lgb.plot_metric(evals_result, metric='binary_logloss')  # metric的值与之前的params里面的值对应
    # auc_dir = os.path.join('./result/muser', "{}_{}.png".format(lgb_model_name, train_data_model))
    # plt.savefig(auc_dir)
    # plt.clf()

    # 保存模型
    # model_dir = os.path.join('./result/muser', "{}_{}_model_newfeature.txt".format(lgb_model_name, train_data_model))
    # print("save model: ", model_dir)
    # n_gbm.save_model(model_dir)
