# -*- coding: utf-8 -*-

"""
入力データの形式について
  1. データはcsvファイル形式で与える。
  2. headerは順に Name(ワーカ名:0~), Prob_No.(問題番号:0~99), ラベル名...,
  3. ラベルはワーカがそのラベルをつけている場合には1, つけていない場合には0を与えている。
"""

import numpy as np
import warnings
import pandas as pd
import sys
from scipy import optimize

# warningは別途処理をしている。
warnings.simplefilter('ignore', RuntimeWarning)

# マルチクラス用のGLAD (Generative model of Labels, Abilities, and Difficulties)
class MultiGLAD:
    def __init__(self,n_labels, n_workers, n_tasks, nu = 1.0):
        """
        (self.)n_labels : 候補ラベル数
        (self.)n_workers : クラウドワーカの数
        (self.)n_tasks : タスクの数
        self.alpha : クラウドワーカの能力 (初期化は原論文に基づく)
        self.beta : タスクの難しさ (初期化は原論文に基づく)
        nu : 正則化パラメータ
        self.prior_z : zの事前確率, 現状「局所一様事前分布を用いている」
        """
        self.n_labels = int(n_labels)
        self.n_workers = int(n_workers)
        self.n_tasks = int(n_tasks)
        self.nu = float(nu)
        self.alpha = np.random.normal(1, 1, n_workers)
        self.beta = np.exp(np.random.normal(1, 1, n_tasks))
        self.prior_z = np.ones(self.n_labels) / self.n_labels

    def get_status(self):
        """
        基本情報を出力する関数
        """
        print("   +-- < data status > --+")
        print("   |   n_labels  : {0:4d}  |".format(self.n_labels))
        print("   |   n_workers : {0:4d}  |".format(self.n_workers))
        print("   |   n_tasks   : {0:4d}  |".format(self.n_tasks))
        print("   +---------------------+")

    def get_ability(self):
        """
        クラウドワーカの能力を出力する関数
        """
        print("   +---- < Ability > ----+")
        for (i, a) in enumerate(self.alpha):
            print("   | No.{0:3d} :  {1:+.6f} |".format(i,a))
        print("   +---------------------+")

    def get_difficulty(self):
        """
        問題の難しさを出力する関数
        """
        print("   +--- < Difficulty > ---+")
        for (i, b) in enumerate(self.beta):
            print("   | No. {0:3d} :  {1:+.6f} |".format(i,b))
        print("   +----------------------+")

    def _mold_data(self, csv_data, expression = False):
        """
        行列形式のcsv_dataを入力として与えて、tensor形式のラベルデータを保持する。
        tensor形式のラベルデータ:
            [クラウドワーカNo., タスクNo., ラベルNo.]でアクセスできる。
            得られていない部分はnanとなっている。
        csv_data : csvファイルを読み取ってnumpy形式にした行列 (詳細は最上部を参照)

        """
        if expression:
            print("Molding data")
        label_matrix = np.zeros([self.n_workers, self.n_tasks])
        label_matrix[:,:] = float("NaN")
        for row in csv_data:
            label_matrix[row[0],row[1]] = row[2]

        if expression:
            print("Finished to Mold data")
        return label_matrix

    # log(sigma(x))
    def _log_sigma(self, x):
        return - np.maximum(0,-x)+np.log(1+np.exp(-np.abs(x)))

    # log((1 / (K - 1)) * (1 - sigma(x)))
    def _Ilog_sigma(self, x):
        return - np.log(self.n_labels - 1) - np.maximum(0,x)+np.log(1+np.exp(-np.abs(x)))

    def _E_step(self):
        # 事前確率 : self.prior_z
        # 尤度 : likelihood
        # 事後確率 : post_z

        # alphaのベクトルとbetaのベクトルを掛けて行列を作成。
        # [alphaの番号, betaの番号]
        x = self.alpha[:, np.newaxis].dot(self.beta[np.newaxis, :])

        # あたりの場合
        log_sigma = self._log_sigma(x)
        # ハズレの場合
        Ilog_sigma =  self._Ilog_sigma(x)

        def compute_likelihood(k):
            likelihood = np.where(self.label_matrix == k, log_sigma, Ilog_sigma)
            likelihood = np.where(np.isnan(self.label_matrix), 0, likelihood)
            return np.exp(likelihood.sum(axis = 0))

        post_z = np.array([compute_likelihood(i) * self.prior_z[i] for i in range(self.n_labels)])
        Z = post_z.sum(axis = 0)
        post_z = (post_z / Z).T
        if np.any(np.isnan(post_z)):
            sys.exit('Error:  Invalid Value [E_step]')
        return post_z

    def _Q_function(self, x):
        # ベクトルの形式で与えられているalphaとbetaを分離する。
        new_alpha = x[:self.n_workers]
        new_beta = x[self.n_workers:]

        Q = (self.post_z * np.log(self.prior_z)).sum()
        # xをここから下は全く別のものとして扱う。
        # alphaのベクトルとbetaのベクトルを掛けて行列を作成。
        # [alphaの番号, betaの番号]
        x = new_alpha[:, np.newaxis].dot(new_beta[np.newaxis, :])
        # nanの処理を後でしなければならない。

        # あたりの場合
        log_sigma = self._log_sigma(x)
        # ハズレの場合
        Ilog_sigma =  self._Ilog_sigma(x)

        def compute_likelihood(k):
            log_likelihood = np.where(self.label_matrix == k, log_sigma, Ilog_sigma)
            log_likelihood = np.where(np.isnan(self.label_matrix), 0, log_likelihood)
            return log_likelihood

        z = np.array([compute_likelihood(i) for i in range(self.n_labels)])
        Q += (self.post_z * z.transpose((1,2,0))).sum()
        # 正則化ペナルティ
        Q -= self.nu * ((new_alpha ** 2).sum() + (new_beta ** 2).sum())
        return Q

    def _MQ(self,x):
        return - self._Q_function(x)

    @staticmethod
    @np.vectorize
    def _sigma(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    @np.vectorize
    def _Isigma(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1.0 - 1e-15
        if x >= sigmoid_range:
            return 1e-15

        return 1.0 / (1.0 + np.exp(x))

    def _gradient(self, x):
        # ベクトルで与えられるalphaとbetaを分ける。
        new_alpha = x[:self.n_workers]
        new_beta = x[self.n_workers:]

        y = new_alpha[:, np.newaxis].dot(new_beta[np.newaxis, :])
        sigma = self._sigma(y)
        Isigma = self._Isigma(y)

        def compute_likelihood(k):
            likelihood = np.where(self.label_matrix == k, Isigma, -sigma)
            likelihood = np.where(np.isnan(self.label_matrix), 0, likelihood)
            return likelihood

        z = np.array([compute_likelihood(i) for i in range(self.n_labels)])
        dQ_dalpha = (self.post_z * (z * new_beta).transpose((1,2,0))).sum(axis = (1,2)) - self.nu * new_alpha
        dQ_dbeta = (self.post_z * (z.transpose((0,2,1)) * new_alpha).transpose((2,1,0))).sum(axis = (0,2)) - self.nu * new_beta
        return np.r_[-dQ_dalpha, -dQ_dbeta]

    def _M_step(self):
        init_params = np.r_[self.alpha, self.beta]
        params = optimize.minimize(fun = self._MQ, x0=init_params, method='CG',jac = self._gradient, tol=0.01,options={'maxiter': 25, 'disp': False})
        self.alpha = params.x[: self.n_workers]
        self.beta = params.x[self.n_workers:]
        return -params.fun

    def _EM_algo(self, tol, max_iter):
        """
        コードの概要
        for
            E_step
            M_step
            収束チェック
        """
        # 事後確率P(Z|l,alpha,beta)を計算する
        self.post_z = self._E_step()
        # 更新前のalphaとbetaを保持する。
        alpha = self.alpha.copy()
        beta = self.beta.copy()
        # 最適化の際に便利なようにalphaとbetaを一つの行列に変換
        x = np.r_[alpha, beta]
        # 現在のQを計算しておく。
        now_Q = self._Q_function(x)
        # 収束するまで更新を繰り返す。
        for i in range(max_iter):
            prior_Q = now_Q
            now_Q = self._M_step()
            self.post_z = self._E_step()
            # 収束判定
            if np.abs((now_Q - prior_Q) / prior_Q) < tol:
                break
        return self.post_z

    def predict(self, data, tol = 0.0001, max_iter = 1000):
        debag = True
        """
        tol : 収束判定に用いる値
        max_iter : iterationの最大回数
        """
        # データの入力
        self.label_matrix = self._mold_data(data)
        # 本クラスのプロパティであるself.post_Zの値をEMアルゴリズムで更新する。
        # EM_algorithm
        # イタレーション回数(max_iter)と収束値(tol)を入れる。
        self._EM_algo(tol, max_iter)
        # 結果出力
        return self.post_z

def main():
    # n_labels：候補ラベル数
    # n_workers：クラウドワーカの数
    # n_tasks：タスクの数
    # filename：データが入ったファイル
    MGLAD = MultiGLAD(n_labels, n_workers, n_tasks)
    Bdata = np.array(pd.read_csv(filename))
    pred = MGLAD.predict(Bdata)

if __name__ == '__main__':
    main()
