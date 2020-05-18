# -*- coding: utf-8 -*-

import sys
import random
import math
import os
import numpy as np
import csv
from operator import itemgetter


random.seed(0)


class UserCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = 900
        self.n_rec_movie = 10

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)


    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        fp = open(filename, 'r')
        for line in fp.readlines():
            user, movie, rating, _ =line.split('\t')
            # split the data by pivot
#            random.random()生成一个[0,1)的函数
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)

    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
#        print ('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
#        print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print ('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
#        print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1
#        print ('build user co-rated movies matrix succ', file=sys.stderr)

        # calculate similarity matrix
#        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
#       计算用户之间的相似性
        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

#        print ('calculate user similarity matrix(similarity factor) succ',
#               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    def recommend(self, user, movie_pre, hit):
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user
        N = 0
        sim_count = 0.0
        rank = dict()
        watched_movies = self.trainset[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(), key=itemgetter(1), reverse=True)[0:K]:
#            找到和当前用户最相似的前K个用户
            rank.setdefault(movie_pre, 0)
            if movie_pre in watched_movies:
                rank[movie_pre] = self.trainset[user][movie_pre]
                continue
            if movie_pre in self.trainset[similar_user]:

                # predict the user's "interest" for each movie
                rank[movie_pre] += similarity_factor*self.trainset[similar_user][movie_pre]
                sim_count += similarity_factor
        
        if sim_count:
            rank[movie_pre] = rank[movie_pre]/sim_count
        else:
            N += 1
#            print('the new movie is %d' % hit, file=sys.stderr)
        # return the N best movies
        return rank[movie_pre]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0
        err = 0.0

#        for i, user in enumerate(self.testset):
        for user, test_movies in self.testset.items():
#            if i % 500 == 0:
#                print ('recommended for %d users' % i, file=sys.stderr)
#            test_movies = self.testset.get(user, {})
            for movie in test_movies:
                movie_pre_rank = self.recommend(user, movie, hit)
                hit+=1
                movie_rank = self.testset[user][movie]
                err += np.power(movie_pre_rank-movie_rank, 2)
                test_count += 1

        rmse = np.sqrt(err/test_count)

        print ('the totle RMSE is = %.6f' % rmse, file=sys.stderr)

if __name__ == '__main__':
    ratingfile = "train.txt"
    usercf = UserCF()
    usercf.generate_dataset(ratingfile)
    usercf.calc_user_sim()
    usercf.evaluate()