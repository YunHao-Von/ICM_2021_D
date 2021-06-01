def fihser_YouxuPanbie(X,k):
    global sampleD, loss
    import numpy as np
    import jieba
    def normalizeRows(xx):  #
        '''
        将矩阵正规化
        x:ndarr
        return:ndarr
        '''
        norm = []

        for i in range(xx.shape[0]):
            mean = np.mean(xx[i])
            std = np.std(xx[i])
            norm.append(1.0 * (xx[i] - mean) / std)
        return (np.array(norm))

    def sampleD(x):
        '''
        计算离差平方和来表示直径
        x:ndarr
        return:number
        '''
        meanVector = np.mean(x, axis=0)
        D = 0
        for i in x:
            D += (meanVector - i) @ (meanVector - i).T
        return D

    def loss(x, interval):
        '''
        x:the normalized matrix,ndarr
        interval:all the classes interval,2 d list
        return:the loss,number
        '''
        loss = 0
        for button, top in interval:
            loss += sampleD(x[button:top])
        #         print(button,top,sampleD(x[button:top]))
        return loss

    class FOP():
        '''
        fisher最优分割'''

        def __init__(self, x, n_classes=4):
            self.LP = np.zeros((x.shape[0], n_classes + 1))  # 可以提前 把我已经训练好的数据放到这
            self.PATH = np.zeros((x.shape[0], n_classes + 1)).tolist()  # 可以提前 把我已经训练好的数据放到这,在../data中

        def FOP_2(self, xx, n_num, n_classes=2, intv=True):
            '''
            find the 2 classes Fisher optimal partition
            xx:the normalized matrix ndarr
            n_num:the length of data
            n_classes:
            intv:decide return the interval or not
            return:the best one classes interval list
                    the min loss,number
            '''
            n_num = n_num - 1

            if self.LP[n_num, n_classes] != 0:  # def a zeros Lp matrix

                return self.PATH[n_num][n_classes], self.LP[n_num, n_classes]

            #     x=normalizeRows(x)
            x = xx.copy()
            x = x[0:n_num + 1]
            length = x.shape[0]
            minLoss = np.inf
            bestInterval = []
            for i in (range(1, length)):
                interval = [[0, i], [i, length]]
                tempLoss = loss(x, interval)
                if tempLoss < minLoss:
                    #             print('find a better one')
                    minLoss = tempLoss
                    bestInterval = interval
            #     print('best',minLoss)
            self.LP[n_num, n_classes] = minLoss
            self.PATH[n_num][n_classes] = bestInterval  # 将路径和loss存储
            if intv:
                return bestInterval, minLoss
            else:
                return minLoss

        def FOP_3(self, xx, n_num, n_classes=3, intv=True):
            '''
            find the 3 classes Fisher optimal partition
            x:the normalized matrix ndarr
            intv:return interval or not
            return:the best one classes interval list
                    the min loss,number
            '''
            n_num = n_num - 1
            if self.LP[n_num, n_classes] != 0:  # def a zeros Lp matrix
                return self.PATH[n_num][n_classes], self.LP[n_num, n_classes]
            x = xx.copy()
            x = x[0:n_num + 1]
            length = x.shape[0]
            minLoss = np.inf
            bestInterval = []
            for i in (range(3 - 1, length)):
                #             print('dangla')
                #             print('I',end='')
                interval = [[0, i], [i, length]]
                #             print(interval)
                _, minLoss_old = self.FOP_2(x, i)
                tempLoss = loss(x, [interval[1]]) + minLoss_old
                #         print(loss(x,[interval[1]]))
                #         print(minLoss_old)
                if tempLoss < minLoss:
                    #             print('find a better one')
                    minLoss = tempLoss
                    bestInterval = interval
                    #     print('best',minLoss)
                    bestInterval[0] = _
            self.LP[n_num, n_classes] = minLoss
            self.PATH[n_num][n_classes] = bestInterval

            #         if n_num > 400 or n_num %10 == 0:
            #             print(n_num,':',bestInterval,end=':')
            if intv:
                return bestInterval, minLoss
            else:
                return minLoss

        def FOP_N(self, xx, n_num, n_classes, intv=True):
            '''
            find the n classes Fisher optimal partition,the n > 2
            x:the normalized matrix ndarr
            intv:return interval or not
            return:the best one classes interval list
                    the min loss,number
            '''
            n_num = n_num - 1
            if self.LP[n_num, n_classes] != 0:  # def a zeros Lp matrix
                return self.PATH[n_num][n_classes], self.LP[n_num, n_classes]
            x = xx.copy()
            x = x[0:n_num + 1]
            length = x.shape[0]
            minLoss = np.inf
            bestInterval = []
            for i in (range(n_classes - 1, length)):
                #             print('dangla')
                # print('I', end='')
                interval = [[0, i], [i, length]]
                #             print(interval)
                if n_classes >= 5:
                    _, minLoss_old = eval('self.FOP_N(x,i,{})'.format(n_classes - 1))
                else:
                    _, minLoss_old = eval('self.FOP_{}(x,i)'.format(n_classes - 1))
                tempLoss = loss(x, [interval[1]]) + minLoss_old
                #         print(loss(x,[interval[1]]))
                #         print(minLoss_old)
                if tempLoss < minLoss:
                    #             print('find a better one')
                    minLoss = tempLoss
                    bestInterval = interval
                    #     print('best',minLoss)
                    bestInterval[0] = _
            self.LP[n_num, n_classes] = minLoss
            self.PATH[n_num][n_classes] = bestInterval

            if n_num > 400 or n_num % 10 == 0:
                yyyyyyyy = 0
                # print(n_num, ':', bestInterval, end=':')
            if intv:
                return bestInterval, minLoss
            else:
                return minLoss
    x=X
    N_CLASS = k
    fop = FOP(x, n_classes=N_CLASS)
    print(fop.FOP_N(x, x.shape[0], n_classes=N_CLASS, intv=True))
