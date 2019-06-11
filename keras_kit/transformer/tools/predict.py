# _*_ coding: utf-8 _*_

import argparse
import os
import time

from ..tag.model import get_or_create


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="执行命令行分词")
    parser.add_argument("-s", "--text", help="要进行分割的语句")
    parser.add_argument("-f", "--file", help="要进行分割的文件。", default="../data/restore.utf8")
    parser.add_argument("-o", "--out_file", help="分割完成后输出的文件。", default="../data/pred_text.utf8")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = get_or_create("../config/default-config.json",
                              src_dict_path="../config/src_dict.json",
                              tgt_dict_path="../config/tgt_dict.json",
                              weights_path="../models/weights.35--0.20.h5")

    text = args.text
    file = args.file
    out_file = args.out_file

    texts = []
    if text is not None:
        start_time = time.time()
        texts = text.split(' ')
        results = tokenizer.decode_texts(texts)
        print(results)
        print(f"cost {(time.time() -start_time) * 1000}ms")

    elif file is not None:
        with open(file, encoding='utf-8') as f:
            texts = list(map(lambda x: x[0:-1], f.readlines()))

        if out_file is not None:
            with open(out_file, mode="w+", encoding="utf-8") as f:
                for text in texts:

                    if len(text) > tokenizer.max_seq_len:
                        if len(max(text.split('。'), key=len)) <= tokenizer.max_seq_len:
                            for t in text.split('。'):
                                if len(t) != 0:
                                    seq, tag = tokenizer.decode_texts([t])[0]
                                    f.write(' '.join(seq) + ' 。 ')
                            f.write("\n")
                        elif len(max(text.split('！'), key=len)) <= tokenizer.max_seq_len:
                            for t in text.split('！'):
                                if len(t) != 0:
                                    seq, tag = tokenizer.decode_texts([t])[0]
                                    f.write(' '.join(seq) + ' ！ ')
                            f.write("\n")
                        elif len(max(text.split('？'), key=len)) <= tokenizer.max_seq_len:
                            for t in text.split('？'):
                                if len(t) != 0:
                                    seq, tag = tokenizer.decode_texts([t])[0]
                                    f.write(' '.join(seq) + ' ？ ')
                            f.write("\n")
                        elif len(max(text.split('，'), key=len)) <= tokenizer.max_seq_len:
                            for t in text.split('，'):
                                if len(t) != 0:
                                    seq, tag = tokenizer.decode_texts([t])[0]
                                    f.write(' '.join(seq) + ' ， ')
                            f.write("\n")
                        elif len(max(text.split('；'), key=len)) <= tokenizer.max_seq_len:
                            for t in text.split('；'):
                                if len(t) != 0:
                                    seq, tag = tokenizer.decode_texts([t])[0]
                                    f.write(' '.join(seq) + ' ； ')
                            f.write("\n")
                        elif len(max(text.split('、'), key=len)) <= tokenizer.max_seq_len:
                            for t in text.split('、'):
                                if len(t) != 0:
                                    seq, tag = tokenizer.decode_texts([t])[0]
                                    f.write(' '.join(seq) + ' 、 ')
                            f.write("\n")
                        else:
                            print("Ignore line: " + text)

                    else:
                        seq, tag = tokenizer.decode_texts([text])[0]
                        f.write(' '.join(seq) + '\n')
