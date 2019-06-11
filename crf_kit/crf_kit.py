# _*_ coding: utf-8 _*_

import os


def run(command):
    os.system(command)


def train(train_file, model_file='model.crf', template_file='template', f=3, c=2):
    params = ['-f', f, '-c', c, template_file, train_file, model_file]
    param = ' '.join(params)
    command = 'crf_learn ' + param
    run(command)


def predict(test_file, model_file='model.crf', target_file='result'):
    params = ['-m', model_file, test_file, '>', target_file]
    param = ' '.join(params)
    command = 'crf_test ' + param
    run(command)


def extract_result(target_file='result'):
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


if __name__ == '__main__':
    target_file = 'entity_full_test.txt'
    ret = extract_result(target_file)
    print(ret)