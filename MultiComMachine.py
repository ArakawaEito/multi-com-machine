import numpy as np
class MultiComMachine:  
    
    ## コンストラクタ
    def __init__(self, parameters_file): #parameters_file:パラメータデータのパス(str)):
        # データ属性の初期化
        self.ClassNum = 1000    #クラス数
        self.machine = [None] #IntegratedComMachineオブジェクトを格納するリスト
        para = Para(parameters_file) #パラメータ(データ属性ではない)
        ComMachine.para = para
        IntegratedComMachine.para = para
     
    ## デストラクタ
    def __del__(self):
        pass
    
    ## 機械初期化
    def initialize(self):
        self.machine.clear()
        self.machine = [None]*(self.ClassNum-1) #IntegratedComMachineオブジェクトを格納するリスト
            
    ## 学習
    def fit(self, x , y):
        self.ClassNum = len(sorted(list(set(y))))  #クラス数
        self.initialize()
        
        for i in range(self.ClassNum-1):
            tmachine = ComMachine(i, x, y) #  2クラス分類多数決機械オブジェクト
            tmachine.training()
            self.machine[i] = IntegratedComMachine(Integrated = tmachine.bestComMachine)
            del tmachine

    ##予測
    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(self.ClassNum-1):
                y_pred[i] = self.machine[j].predict(x[i])  #self.machine[j]:IntegratedComMachine
                if y_pred[i] == j:
                    break
        return y_pred

### 2クラス分類多数決機械オブジェクトのクラス
### クラスがclassNoであるデータとクラスがclassNoよりも大きいデータに分類する
class ComMachine:
    # クラス属性（静的データメンバ）の初期化
    para = None
    
    ## コンストラクタ
    def __init__(self, aclassNo, x, y):
        
        # データ属性の初期化
        self.machineNum = 1                                   # 線形機械の個数
        
        ##aclassNo以上のデータだけを抽出する
        index_classnum = np.where(y >= aclassNo)        
        self.data = x[index_classnum]
        self.target = y[index_classnum]
        self.AttNum = self.data.shape[1]                            #属性数
        self.allDataNum = self.data.shape[0]                        #データ数
        self.data = np.insert(self.data, self.AttNum, 1, axis = 1)  #データの最後の列に１の列を追加（valueを計算しやすくするため）
        self.classNo = aclassNo        
        self.comptarget = np.where(self.target==self.classNo, 1, 0)          #fix関数で使用
        
        #各線形機械の内積値を統合したvalue行列の初期化．行数がデータ件数，列数が線形機械の個数
        self.value_integrated = np.zeros((self.allDataNum, self.machineNum), dtype = 'float64')
        
        #各線形機械の判定結果を統合したresult行列の初期化．行数がデータ件数，列数が線形機械の個数
        self.result_integrated = np.where(self.target[:, np.newaxis] == self.classNo, 1, -1)  # 正解：1，不正解：-1, [:, np.newaxis]:一次元から二次元に変換
        self.index_true = np.where(self.target == self.classNo)  # convergence関数で使用      
   
        #各線形機械の重みを統合したweight行列の初期化．行数が属性数+1，列数が線形機械の個数
        self.weight_integrated = np.zeros((self.AttNum+1, self.machineNum), dtype = 'float64')
        
        #各行を合計することで多数決（正なら正解，負なら不正解）
        self.result = np.sum(self.result_integrated, axis=1)
        self.correct = np.count_nonzero(self.result > 0 )    # 正解訓練データ数   

        self.conv = 0                                            #学習反復回数        
        self.bestComMachine = IntegratedComMachine(ComMachine = self)    #2クラス分類多数決機械（統合型）オブジェクト．統合型：線形機械を合わせて1つのオブジェクトにしたもの
        
    ## デストラクタ
    def __del__(self):
        pass
    
    ##学習
    def training(self):
        import itertools
        maxcorrect = -1
        ## 関数のオブジェクトを変数に代入（少しでも処理を速くするため）
        test = self.test
        nextData = self.nextData
        fix = self.fix
        convergence = self.convergence
        
        allDataNum=self.allDataNum
        RepNumMax=ComMachine.para.RepNumMax
        
        # 線形機械の数が最大値になるorすべての訓練データが正解訓練データになる,まで繰り返す
        for i in itertools.count():                   #無限ループ
            dataNo = -1
            test()
            self.conv = 0    
            for i in itertools.count():               #無限ループ
                dataNo = nextData(dataNo) # 不正解となった訓練データの添え字を特定する
                fix(dataNo)               # 多数決機械の出力と訓練データのクラスが一致しなかったときに，識別を誤った線形機械のパラメータの修正
                if maxcorrect < self.correct:
                    self.bestComMachine.renewal(self)
                    maxcorrect = self.correct              
                self.conv += 1
                if (self.correct == allDataNum) or (self.conv >= RepNumMax):
                    break         
            if (self.correct == allDataNum) or (convergence()):
                break  
                
    ## dataNo以降の不正解の箇所を探す
    ## dataNo : データ番号
#     def nextData(self, dataNo):
#         index_false = np.where(self.result < 0)[0]  # 判別を誤ったデータ番号の配列を取得
#         next_index_false = index_false[np.where(index_false > dataNo)]
#         if len(next_index_false) !=0:
#             return next_index_false[0]
#         else:
#             return  index_false[0]
    
    def nextData(self, dataNo):
        for i in range(dataNo + 1, self.allDataNum):
            if self.result[i] <0:
                return i
        for i in range(dataNo+1):
            if self.result[i] <0:
                return i            
            
    ## テスト
    def test(self):
        self.correct = 0
        #各行を合計することで多数決（正なら正解，負なら不正解）
        self.result = np.sum(self.result_integrated, axis=1)
        self.correct = np.count_nonzero(self.result > 0 )    # 正解訓練データ数   
    
    ## 重み修正
    def fix(self, dataNo):
        # 内積の絶対値が最小の機械を探す 
        false_machine_index = np.where(self.result_integrated[dataNo] == -1)[0]  #識別を誤った線形機械のインデックスリスト
        abs_value = np.abs(self.value_integrated[dataNo,false_machine_index]) 
        min_value_index = abs_value.argmin()     
        min_value_machine_index = false_machine_index[min_value_index] # 内積の絶対値が最小の機械のインデックス
 
        #　重み変化係数
        coef1 = ComMachine.para.Cofficient
        coef2 = ComMachine.para.Cofficient2  # 絶対値が最小の時
        # 重み変化係数のリスト.識別を誤っていない線形機械に対して０を割り当てる
        index_machine = np.array(range(self.machineNum))
        coef = np.where((self.result_integrated[dataNo] == -1)&(self.value_integrated[dataNo] >= 0), -1, 0)
        coef[(self.result_integrated[dataNo] == -1)&(self.value_integrated[dataNo] < 0)] = 1
        coef = np.where(index_machine == min_value_machine_index, coef2*coef, coef1*coef)
        
        # 重み修正
        self.weight_integrated+= coef*self.data[dataNo][:, np.newaxis]     
        
        ##データと重みの内積値を計算
        self.value_integrated = np.dot(self.data, self.weight_integrated) 
        
        #self.value_integratedの符号情報を格納（正なら0，非正なら1）
        sign_value = np.where(self.value_integrated>=0, 0, 1)
        # 判別結果をself.result_integratedに格納
        comp = np.tile(self.comptarget[:, np.newaxis], self.machineNum)
        self.result_integrated = np.where(comp^sign_value  == 1, 1, -1) # ^ : 排他的論理和をとることで下の処理を高速化
##################################################################################
#         for i in range(self.allDataNum):
#             for j in range(self.machineNum):              
#                 if (self.value_integrated[i, j] >= 0 and self.target[i] ==  self.classNo) or (self.value_integrated[i, j] <  0 and self.target[i] !=  self.classNo):
#                     self.result_integrated[i, j] = 1
#                 else:
#                     self.result_integrated[i, j] = -1  
####################################################################################

        #クラス判定結果と正解データ数.各行を合計することで多数決（正なら正解，負なら不正解）
        self.result = np.sum(self.result_integrated, axis=1)
        self.correct = np.count_nonzero(self.result > 0 )    # 正解データ数          
    
    ## 収束判定．線形機械の数を2個ずつ増やす.
    def convergence(self):
        if self.machineNum < ComMachine.para.MachineNumMax -1:
            self.value_integrated = np.insert(self.value_integrated, self.machineNum, 0.0, axis = 1) 
            self.weight_integrated = np.insert(self.weight_integrated, self.machineNum, 0.0, axis = 1) 
            self.result_integrated = np.insert(self.result_integrated, self.machineNum, -1, axis = 1) 
            self.result_integrated[self.index_true, self.machineNum] = 1
            self.machineNum += 1
            
            self.value_integrated = np.insert(self.value_integrated, self.machineNum, 0.0, axis = 1) 
            self.weight_integrated = np.insert(self.weight_integrated, self.machineNum, 0.0, axis = 1) 
            self.result_integrated = np.insert(self.result_integrated, self.machineNum, -1, axis = 1) 
            self.result_integrated[self.index_true, self.machineNum] = 1
            self.machineNum += 1         
            return False
        else:
            return True
            

### 2クラス分類多数決機械（統合型）オブジェクトのクラス
### 統合型：線形機械を合わせて1つのオブジェクトにしたもの
class IntegratedComMachine:
    # クラス属性（静的データメンバ）の定義
    para = None
    
    ## コンストラクタ
    def __init__(self, ComMachine = False, Integrated = False):
        if Integrated == False:
            self.machineNum = ComMachine.machineNum
            self.AttNum = ComMachine.AttNum            #属性数
            self.allDataNum = ComMachine.allDataNum    #データ数
            
            self.weight = ComMachine.weight_integrated.copy()
            self.classNo = ComMachine.classNo
            self.conv = ComMachine.conv
        else:
            self.machineNum = Integrated.machineNum
            self.AttNum = Integrated.AttNum            #属性数
            self.allDataNum = Integrated.allDataNum    #データ数 
            self.weight = Integrated.weight
 
            self.classNo = Integrated.classNo
            self.conv = Integrated.conv
 
    ## デストラクタ
    def __del__(self):
        pass
    
    ## ComMachineをコピー
    def renewal(self, model):
        self.machineNum = model.machineNum
        self.weight = model.weight_integrated.copy() 
        self.conv = model.conv        
        
    ## 予測
    def predict(self, x):
        opinion = 0
        x = x[np.newaxis, :]
        x = np.insert(x, self.AttNum, 1, axis = 1)     #データの最後の列に１の列を追加（valueを計算しやすくするため）       
        value = np.dot(x, self.weight)    
    
        # valueの符号情報を格納（正なら1，非正なら-1）
        sign_value = np.where(value>=0, 1, -1)
        # 各行を合計することで多数決（正ならclassNo，負ならclassNo+1）
        opinion = np.sum(sign_value, axis=1)
        if opinion > 0:
            return self.classNo
        else:
            return self.classNo+1
        
            
### パラメータ格納用オブジェクトのクラス
class Para:
    
    ## コンストラクタ
    def __init__(self, parameters_file):# parameters_file:パラメータデータのパス(str)
        import pandas as pd
        ## パラメータファイル（csv）の読み込み
        try:
            parameters = pd.read_csv(parameters_file,  encoding = 'utf_8').iloc[0]
        except Exception as e:
            print(f'Cannot open {os.path.basename(parameters_file)} for output\n')
            sys.exit()
            
        # データ属性の初期化
        self.Cofficient = float(parameters.loc["Cofficient"])
        self.Cofficient2 = float(parameters.loc["Cofficient2"])
        self.RepNumMax = int(parameters.loc["RepNumMax"])
        self.MachineNumMax = int(parameters.loc["MachineNumMax"])
        self.CrossValNum = int(parameters.loc["CrossValNum"])
    
    
    ## デストラクタ
    def __del__(self):
        pass
