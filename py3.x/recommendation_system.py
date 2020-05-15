from numpy import *
from random import *
import csv


def process_data(trainDat, movieDat):
    user_type = zeros((943, 18))  # 创建一个用户-电影类型矩阵，其中每一行代表一个用户看过的不同种类的电影的数量,维度为943*18
    user_score = zeros((943, 1682))  # 创建一个用户-电影得分矩阵，每一行是每个用户对所有电影的打分值，维度为943*1682
    movie_user = []  # 创建一个电影-用户list，来记录对每个电影，看过的此电影的用户
    for i in range(1682):
        movie_user.append([])
    for i in range(len(trainDat)):
        movie_user[trainDat[i][1] - 1].append(trainDat[i][0])
        user_score[trainDat[i][0] - 1][trainDat[i][1] - 1] = int(trainDat[i][2])
        for j in range(18):
            user_type[trainDat[i][0] - 1][j] += int(movieDat[trainDat[i][1] - 1][j])
    return user_type, user_score, movie_user


def Similarity(type_list1,type_list2):
    if (type_list1==type_list2).sum() == 0:  # 如果两个用户没有任何都看过的电影类型，则相似度为0
        return 0
    else:
        return 1/(1+sqrt(sum((type_list1-type_list2)**2)))  # 欧式距离计算用户相似度


def RMSE(prediction,test_data):
    squre_sum=0.0
    for i in range(len(test_data)):
        squre_sum+=(prediction[i]-test_data[i][2])**2
    rmse=sqrt(squre_sum/len(prediction))
    return rmse


def predict(test_data,movie_user,user_score,user_type):
    prediction=[]
    for i in range(len(test_data)):
        user=int(test_data[i][0])  # 待测用户
        movie=int(test_data[i][1])  # 待测电影
        prediction_i=0.0
        if sum(user_type[user-1])==0:  # 如果待测用户从未看过任何类型电影，则取预测得分值为1-5的均值3.0
            prediction_i=3.0
        else:
            if len(movie_user[movie-1])==0:  # 如果没有人看过待测电影，则取待测用户的历史得分的均值为预测得分
                prediction_i= sum(user_score[user-1])/len(user_score[user-1])
            else:
                sim=zeros(len(movie_user[movie-1]))  # 求出此待测用户和所有对该待测电影评过分的用户的相似度
                numerator=0.0
                for j in range(len(movie_user[movie-1])):
                    sim[j]=Similarity(user_type[user-1],user_type[movie_user[movie-1][j]-1])
                    numerator+=sim[j]*user_score[movie_user[movie-1][j]-1][movie-1]
                if sim.sum()==0: # 如果所有看过该电影的用户和该用户的相似度都为0，则取待测用户的历史得分的均值为预测得分
                    prediction_i=sum(user_score[user-1])/len(user_score[user-1])
                else:  # 不然则按公式计算
                    prediction_i=numerator/sum(sim)
        prediction_i=round(prediction_i)
        prediction.append(prediction_i)
    return prediction


def check_RMSE(train_data, movie_set):
    print('RMSE on the check_dataset:')
    train_data_copy = train_data.copy()
    shuffle(train_data_copy)
    check_dataset=train_data_copy[0:9057]  # 留出法，留出10%计算RMSE对算法进行评估
    train_dataset=train_data_copy[9057:]  # 剩下的留作训练集
    user_type, user_movie, movie_user = process_data(train_dataset, movie_set)
    prediction = predict(check_dataset, movie_user, user_movie, user_type)
    rmse=RMSE(prediction,check_dataset)  # 在留出的check_dataset上计算RMSE
    print('RMSE = ' + str(rmse))


train_data_path,movie_set_path,test_data_path,prediction_path= 'train.txt','movie_genres.txt','test.txt','prediction.csv'
train_data,movie_set,test_data=[],[],[]
for line in open(movie_set_path,encoding='ISO-8859-1').readlines():
    line_list=line.split('|')
    movie_set.append(line_list[6:])  # 从第6维开始存，前面4维不是type，第5维unknown也忽略不计
for line in open(train_data_path).readlines():
    line_list=line.split('\t')
    train_data.append([int(line_list[0]),int(line_list[1]),int(line_list[2])])
for line in open(test_data_path):
    line_list=line.split('\t')
    test_data.append([int(line_list[0]),int(line_list[1])])


def main(train_data,movie_set,test_data,prediction_path):
    check_RMSE(train_data,movie_set)
    print('Making prediction,please wait')
    user_type, user_score, movie_user=process_data(train_data,movie_set)
    prediction=predict(test_data,movie_user,user_score,user_type)
    with open(prediction_path,'w',newline='') as f:
        for i in range(len(test_data)):
            prediction_line=[[int(test_data[i][0]),int(test_data[i][1]),prediction[i]]]
            csv.writer(f).writerows(prediction_line)
    print('Finished!')

if __name__ =='__main__':
    main(train_data,movie_set,test_data,prediction_path)