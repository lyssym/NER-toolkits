# _*_ coding: utf-8 _*_

from ..tag.model import get_or_create


if __name__ == '__main__':
    segmenter = get_or_create("data/default-config.json",
                              src_dict_path="data/src_dict.json",
                              tgt_dict_path="data/tgt_dict.json",
                              weights_path="model/weights.50--0.17.h5")

    texts = [
        "巴纳德星的名字起源于一百多年前一位名叫爱德华·爱默生·巴纳德的天文学家。",
        "他发现有一颗星在夜空中划过的速度很快，这引起了他极大的注意。",
        "新西兰总统来华大基因进行访问，关于基因测序的细节进行了沟通与商谈",
        "印度尼西亚国家抗灾署此前发布消息证实，印尼巽他海峡附近的万丹省当地时间22号晚遭海啸袭击。"
    ]

    for sent, tag in segmenter.decode_texts(texts):
        print(sent)
        print(tag)
