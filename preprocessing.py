import csv
import numpy as np
import re
import pandas as pd

class PreProcessing:
    def __init__(self):
        self.r1 = "[\s+\.\!\-\?\/_,$%^*(+\"]+|[+——！:，。？、~@#￥%……&*（）]+"
        self.r2 = '(\s\'+\s)|(\'(\s|$))|\)'
        self.node2pap = {}
        self.text_id = {}
        self.train = []
        self.test = []

    def load_data(self):
        with open('./data/nodeid2paperid.csv', 'r') as n:
            fn = csv.reader(n)
            header = next(fn)
            for line in fn:
                self.node2pap.update({line[0]: line[1]})

        with open('./data/text.csv', 'r', encoding='utf-8') as t:
            ft = csv.reader(t)
            for line in ft:
                self.text_id.update({line[0]: line[1:]})

        with open('./data/train.csv', 'r') as f:
            ff = csv.reader(f)
            for line in ff:
                tag = line[0]
                pid = line[1]
                text = self.text_id[self.node2pap[pid]]
                text = ' '.join(text)
                self.train.append([text, tag, pid])
        with open('./data/test.csv', 'r') as m:
            fm = csv.reader(m)
            for line in fm:
                tid = line[0]
                t_text = self.text_id[self.node2pap[tid]]
                t_text = ' '.join(t_text)
                self.test.append([t_text, tid])

    def process_train(self, file):
        data = np.array(self.train)
        text = data[:, 0]
        label = data[:, 1]
        trid = data[:, 2]
        for i in range(len(text)):
            test = text[i]
            result = re.sub(self.r1, ' ', test)
            result = re.sub(self.r2, ' ', result)
            result = re.sub('\d+', ' ', result)
            result = re.sub(r'\\', ' ', result)
            result = re.sub('[^a-zA-Z]', ' ', result)
            result = re.sub('\s+', ' ', result)
            text[i] = result
        with open(file, 'w', encoding='utf-8') as wf:
            wf.write('text,label\n')
            for i in range(len(text)):
                wf.write(text[i] + ',' + label[i] + ',' + trid[i] + '\n')

    def process_test(self, file):
        for i in range(len(self.test)):
            test = self.test[i][0]
            result = re.sub(self.r1, ' ', test)
            result = re.sub(self.r2, ' ', result)
            result = re.sub('\d+', ' ', result)
            result = re.sub(r'\\', ' ', result)
            result = re.sub('[^a-zA-Z]', ' ', result)
            result = re.sub('\s+', ' ', result)
            self.test[i][0] = result
        with open(file, 'w', encoding='utf-8') as pf:
            pf.write('test_text,id\n')
            for line in self.test:
                pf.write(line[0] + ',' + line[1] + '\n')

    def process_text(self,text):
        result = re.sub(self.r1, ' ', text)
        result = re.sub(self.r2, ' ', result)
        result = re.sub('\d+', ' ', result)
        result = re.sub(r'\\', ' ', result)
        result = re.sub('[^a-zA-Z]', ' ', result)
        result = re.sub('\s+', ' ', result)
        return result

if __name__ == '__main__':
    df = pd.read_csv('./data/test.csv')
    df.columns = ['label','text']
    pp = PreProcessing()
    clean_text = list(map(pp.process_text,df.text))
    labels = df.label.to_numpy()
    with open('./data/clean.csv','w',encoding='utf-8') as f:
        f.write('text,label\n')
        for i in range(len(clean_text)):
            f.write(str(labels[i])+","+clean_text[i]+'\n')
           # f.write(clean_text[i]+","+str(labels[i])+'\n')


