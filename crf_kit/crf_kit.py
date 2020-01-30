# _*_ coding: utf-8 _*_

import os


def run(command):
    '''
    执行命令
    :param command:
    :return:
    '''
    os.system(command)


def train(train_file, model_file='model.crf', template_file='crf_template', f=3, c=2):
    params = ['-f', str(f), '-c', str(c), template_file, train_file, model_file]
    param = ' '.join(params)
    command = 'crf_learn ' + param
    run(command)


def predict(test_file, model_file='model.crf', target_file='result.txt'):
    params = ['-m', model_file, test_file, '>', target_file]
    param = ' '.join(params)
    command = 'crf_test ' + param
    run(command)


def extract_result(target_file='result.txt'):
    words = []
    labels = []
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            arr = line.split(' ')
            words.append(arr[0].strip())
            labels.append(arr[-1].strip())

    ret = {}
    name = None
    for i in range(len(labels)):
        label = labels[i]
        if label != 'O':
            if name == label:         # 连续标签
                size += 1
            else:
                start = i
                size = 0

                if name is not None:  # 另起连续标签
                    ret[name] = ''.join(words[start:start+size])

                name = label
        else:
            if name is not None:      # 截断提取标签
                ret[name] = ''.join(words[start:start+size])

            start = -1
            size = 0
            name = None

    return ret


def generate_template(template_file, field_count, *counts):
    '''
    根据自定义特征，生成相应的特征模板
    :param template_file:
    :param field_count:
    :param counts:
    :return:
    '''
    wr = open(template_file, 'w', encoding='utf-8')
    feature = "# Unigram Features"
    wr.write(feature + '\n')

    for i in range(field_count):
        feature = "U" + str(i)
        if i == 0:
            count = counts[0]
        else:
            count = counts[1]

        for j in range(2 * count + 1):
            feature += str(j) + ":%x"
            feature += "[" + str(j - count) + "," + str(i) + "]"
            wr.write(feature + '\n')

            feature = "U" + str(i)

    feature = "# Bigram Features"
    wr.write(feature + "\n")
    feature = "B"
    wr.write(feature + "\n")
    wr.close()


def crf(train_file, test_file, template_file='crf_template', features_size=4):
    generate_template(template_file, features_size, 3, 2)
    train(train_file, template_file=template_file)
    predict(test_file)


def fusion_feature(text):
    '''
    形如： 身体部位 与 词性 的融合
    :return:
    '''
    pass


def cascade_crf(train_file, test_file, pre_template_file, pre_features_size, post_template_file, post_features_size):
    generate_template(pre_template_file, pre_features_size, 3, 2)
    train(train_file, template_file=pre_template_file)
    predict(train_file)

    fusion_feature()  # 特征融合

    generate_template(post_template_file, post_features_size, 3, 2)
    train(train_file, template_file=post_template_file)
    predict(test_file)


if __name__ == '__main__':
    # target_file = 'entity_full_test.txt'
    # ret = extract_result(target_file)
    # print(ret)

    # train_file = 'train.txt'
    # generate_template('crf_template', 4, 3, 2)
    # train(train_file)
    # test_file = 'test.txt'
    # predict(test_file)

    train_file = 'train0.txt'
    generate_template('crf_template_disease', 2, 3, 2)
    train(train_file, template_file='crf_template_disease')
    test_file = 'test0.txt'
    predict(test_file)
