import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Bernoulli
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.manifold import TSNE
from utils import *


class emb_model(object):
    def __init__(self, args, d, logdir):
        self.args = args

        self.K = args.K
        self.cs = args.cs
        self.ns = args.ns
        self.sig = args.sig
        self.dynamic = args.dynamic
        self.logdir = logdir
        self.N = d.N
        self.L = d.L
        self.T = d.T
        self.n_minibatch = d.n_train
        self.n_test = d.n_test
        self.labels = d.labels
        self.unigram = d.unigram
        self.dictionary = d.dictionary
        self.query_words = d.query_words
        self.train_feed = d.train_feed
        #self.valid_data = d.valid_data
        #self.test_data = d.test_data
        self.n_iter = args.n_iter
        self.n_epochs = d.n_epochs
        self.n_test = d.n_test
        self.n_valid = d.n_valid
        self.alpha_trainable = True
        if args.init:
            fname = os.path.join('fits', d.name, args.init)
            if 'alpha_constant' in args.init:
                self.alpha_trainable = False
                fname = fname.replace('/alpha_constant','')
            fit = pickle.load(open(fname))
            self.rho_init = fit['rho']
            self.alpha_init = fit['alpha']
        else:
            self.rho_init = (np.random.randn(self.L, self.K)/self.K).astype('float32')
            self.alpha_init = (np.random.randn(self.L, self.K)/self.K).astype('float32')
        if not self.alpha_trainable:
            self.rho_init = (0.1*np.random.randn(self.L, self.K)/self.K).astype('float32')

        with open(os.path.join(self.logdir,  "log_file.txt"), "a") as text_file:
            text_file.write(str(self.args))
            text_file.write('\n')

    def dump(self, fname):
        raise NotImplementedError()

    def detect_drift(self):
        raise NotImplementedError()

    def eval_log_like(self, feed_dict):
        return self.sess.run(tf.log(self.y_pos.mean()+0.000001), feed_dict = feed_dict)


    def plot_params(self, plot_only=500):
        with self.sess.as_default():
	    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_alpha2 = tsne.fit_transform(self.alpha.eval()[:plot_only])
            plot_with_labels(low_dim_embs_alpha2[:plot_only], self.labels[:plot_only], self.logdir + '/alpha.eps')

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_rho2 = tsne.fit_transform(self.rho.eval()[:plot_only])
            plot_with_labels(low_dim_embs_rho2[:plot_only], self.labels[:plot_only], self.logdir + '/rho.eps')

    def print_word_similarities(self, words, num):
        with self.sess.as_default():
            rho = self.rho.eval()
        for x in words:
            x_idx = self.dictionary[x]
            f_name = os.path.join(self.logdir, '%s_queries.txt' % (x))
            with open(f_name, "w+") as text_file:
                cos = cosine_distance(rho[x_idx], rho.T)
                rank = np.argsort(cos)[1:num+1]
                text_file.write("\n\n=====================================\n%s\n=====================================" % (x))
                for r in rank:
                    text_file.write("\n%-20s %6.4f" % (self.labels[r], cos[r]))

    def initialize_training(self):
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()

        variable_summaries('alpha', self.alpha)
        with tf.name_scope('objective'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('priors', self.log_prior)
            tf.summary.scalar('ll_pos', self.ll_pos)
            tf.summary.scalar('ll_neg', self.ll_neg)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        alpha = config.embeddings.add()
        alpha.tensor_name = 'model/embeddings/alpha'
        alpha.metadata_path = '../vocab.tsv'
        if not self.dynamic:
            rho = config.embeddings.add()
            rho.tensor_name = 'model/embeddings/rho'
            rho.metadata_path = '../vocab.tsv'
        else:
            for t in range(self.T):
                rho = config.embeddings.add()
                rho.tensor_name = 'model/embeddings/rho_'+str(t)
                rho.metadata_path = '../vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    
    def train_embeddings(self):
        for data_pass in range(self.n_iter):
            for step in range(self.n_epochs):
                if step % 100 == 0:
                    summary, ll_pos, ll_neg, _ = self.sess.run([self.summaries, self.ll_pos, self.ll_neg, self.train], feed_dict=self.train_feed(self.placeholders))
                    self.train_writer.add_summary(summary, data_pass*(self.n_epochs) + step)
                    print("%8d/%8d iter%3d; log-likelihood: %6.4f on positive samples,%6.4f on negative samples " % (step, self.n_epochs, data_pass, ll_pos, ll_neg))
                else:
                    self.sess.run([self.train], feed_dict=self.train_feed(self.placeholders))
            self.dump(self.logdir+"/variational"+str(data_pass)+".dat")
            self.saver.save(self.sess, os.path.join(self.logdir, "model.ckpt"), data_pass)

        self.print_word_similarities(self.query_words, 10)
        if self.dynamic:
            words = self.detect_drift()
            self.print_word_similarities(words[:10], 10)
        self.plot_params(500)


class bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(bern_emb_model, self).__init__(args, d, logdir)
        self.n_minibatch = self.n_minibatch.sum()

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.placeholders = tf.placeholder(tf.int32)
                self.words = self.placeholders
            

            # Index Masks
            with tf.name_scope('context_mask'):
                self.p_mask = tf.cast(tf.range(self.cs/2, self.n_minibatch + self.cs/2),tf.int32)
                rows = tf.cast(tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch, 1]),tf.int32)
                columns = tf.cast(tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch), [1]), [1, self.cs/2]),tf.int32)
                self.ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

            with tf.name_scope('embeddings'):
                self.rho = tf.Variable(self.rho_init, name='rho')
                self.alpha = tf.Variable(self.alpha_init, name='alpha', trainable=self.alpha_trainable)

                with tf.name_scope('priors'):
                    prior = Normal(loc = 0.0, scale = self.sig)
                    if self.alpha_trainable:
                        self.log_prior = tf.reduce_sum(prior.log_prob(self.rho) + prior.log_prob(self.alpha))
                    else:
                        self.log_prior = tf.reduce_sum(prior.log_prob(self.rho))

            with tf.name_scope('natural_param'):
                # Taget and Context Indices
                with tf.name_scope('target_word'):
                    self.p_idx = tf.gather(self.words, self.p_mask)
                    self.p_rho = tf.squeeze(tf.gather(self.rho, self.p_idx))
                
                # Negative samples
                with tf.name_scope('negative_samples'):
                    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(self.unigram)), [0]), [self.n_minibatch, 1])
                    self.n_idx = tf.multinomial(unigram_logits, self.ns)
                    self.n_rho = tf.gather(self.rho, self.n_idx)

                with tf.name_scope('context'):
                    self.ctx_idx = tf.squeeze(tf.gather(self.words, self.ctx_mask))
                    self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)


                # Natural parameter
                ctx_sum = tf.reduce_sum(self.ctx_alphas,[1])
                self.p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(self.p_rho, ctx_sum),-1),1)
                self.n_eta = tf.reduce_sum(tf.multiply(self.n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
            
            # Conditional likelihood
            self.y_pos = Bernoulli(logits = self.p_eta)
            self.y_neg = Bernoulli(logits = self.n_eta)

            self.ll_pos = tf.reduce_sum(self.y_pos.log_prob(1.0)) 
            self.ll_neg = tf.reduce_sum(self.y_neg.log_prob(0.0))

            self.log_likelihood = self.ll_pos + self.ll_neg
            
            scale = 1.0*self.N/self.n_minibatch
            self.loss = - (self.n_epochs * self.log_likelihood + self.log_prior)


    def dump(self, fname):
            with self.sess.as_default():
              dat = {'rho':  self.rho.eval(),
                     'alpha':  self.alpha.eval()}
            pickle.dump( dat, open( fname, "a+" ) )



class dynamic_bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(dynamic_bern_emb_model, self).__init__(args, d, logdir)

        with tf.name_scope('model'):
            with tf.name_scope('embeddings'):
                self.alpha = tf.Variable(self.alpha_init, name='alpha', trainable=self.alpha_trainable)

                self.rho_t = {}
                for t in range(-1,self.T):
                    self.rho_t[t] = tf.Variable(self.rho_init 
                        + 0.001*tf.random_normal([self.L, self.K])/self.K,
                        name = 'rho_'+str(t))

                with tf.name_scope('priors'):
                    global_prior = Normal(loc = 0.0, scale = self.sig)
                    local_prior = Normal(loc = 0.0, scale = self.sig/100.0)

                    self.log_prior = tf.reduce_sum(global_prior.log_prob(self.alpha))
                    self.log_prior += tf.reduce_sum(global_prior.log_prob(self.rho_t[-1]))
                    for t in range(self.T):
                        self.log_prior += tf.reduce_sum(local_prior.log_prob(self.rho_t[t] - self.rho_t[t-1])) 

            with tf.name_scope('likelihood'):
                self.placeholders = {}
                self.y_pos = {}
                self.y_neg = {}
                self.ll_pos = 0.0
                self.ll_neg = 0.0
                for t in range(self.T):
                    # Index Masks
                    p_mask = tf.range(self.cs/2,self.n_minibatch[t] + self.cs/2)
                    rows = tf.tile(tf.expand_dims(tf.range(0, self.cs/2),[0]), [self.n_minibatch[t], 1])
                    columns = tf.tile(tf.expand_dims(tf.range(0, self.n_minibatch[t]), [1]), [1, self.cs/2])
                    
                    ctx_mask = tf.concat([rows+columns, rows+columns +self.cs/2+1], 1)

                    # Data Placeholder
                    self.placeholders[t] = tf.placeholder(tf.int32, shape = (self.n_minibatch[t] + self.cs))

                    # Taget and Context Indices
                    p_idx = tf.gather(self.placeholders[t], p_mask)
                    ctx_idx = tf.squeeze(tf.gather(self.placeholders[t], ctx_mask))
                    
                    # Negative samples
                    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(self.unigram)), [0]), [self.n_minibatch[t], 1])
                    n_idx = tf.multinomial(unigram_logits, self.ns)

                    # Context vectors
                    ctx_alphas = tf.gather(self.alpha, ctx_idx)

                    p_rho = tf.squeeze(tf.gather(self.rho_t[t], p_idx))
                    n_rho = tf.gather(self.rho_t[t], n_idx)

                    # Natural parameter
                    ctx_sum = tf.reduce_sum(ctx_alphas,[1])
                    p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(p_rho, ctx_sum),-1),1)
                    n_eta = tf.reduce_sum(tf.multiply(n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,self.ns,1])),-1)
                    
                    # Conditional likelihood
                    self.y_pos[t] = Bernoulli(logits = p_eta)
                    self.y_neg[t] = Bernoulli(logits = n_eta)

                    self.ll_pos += tf.reduce_sum(self.y_pos[t].log_prob(1.0)) 
                    self.ll_neg += tf.reduce_sum(self.y_neg[t].log_prob(0.0))

            self.loss = - (self.n_epochs * (self.ll_pos + self.ll_neg) + self.log_prior)

    def dump(self, fname):
            with self.sess.as_default():
                dat = {'alpha':  self.alpha.eval()}
                for t in range(self.T):
                    dat['rho_'+str(t)] = self.rho_t[t].eval()
            pickle.dump( dat, open( fname, "a+" ) )

    def eval_log_like(self, feed_dict):
        log_p = np.zeros((0,1))
        for t in range(self.T):
            log_p_t = self.sess.run(tf.log(self.y_pos[t].mean()+0.000001), feed_dict = feed_dict)
            log_p = np.vstack((log_p, log_p_t))
        return log_p


    def plot_params(self, plot_only=500):
        with self.sess.as_default():
	    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_alpha = tsne.fit_transform(self.alpha.eval()[:plot_only])
            plot_with_labels(low_dim_embs_alpha[:plot_only], self.labels[:plot_only], self.logdir + '/alpha.eps')
            for t in [0, int(self.T/2), self.T-1]:
                w_idx_t = range(plot_only)
                np_rho = self.rho_t[t].eval()
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                low_dim_embs_rho = tsne.fit_transform(np_rho[w_idx_t,:])
                plot_with_labels(low_dim_embs_rho, self.labels[w_idx_t], self.logdir + '/rho_' + str(t) + '.eps')

    def detect_drift(self, metric='total_dist'):
        if metric == 'total_dist':
            tf_dist, tf_w_idx = tf.nn.top_k(tf.reduce_sum(tf.square(self.rho_t[self.T-1]-self.rho_t[0]),1), 500)
        else:
            print('unknown metric')
            return
        dist, w_idx = self.sess.run([tf_dist, tf_w_idx])
        words = self.labels[w_idx]
        f_name = self.logdir + '/top_drifting_words.txt'
        with open(f_name, "w+") as text_file:
           for (w, drift) in zip(w_idx,dist):
               text_file.write("\n%-20s %6.4f" % (self.labels[w], drift))
        return words


    def print_word_similarities(self, words, num):
        for x in words:
            x_idx = self.dictionary[x]
            f_name = os.path.join(self.logdir, '%s_queries.txt' % (x))
            with open(f_name, "w+") as text_file:
                for t_idx in xrange(self.T):
                    with self.sess.as_default():
                        rho_t = self.rho_t[t_idx].eval()
                    cos = cosine_distance(rho_t[x_idx], rho_t.T)
                    rank = np.argsort(cos)[1:num+1]
                    text_file.write("\n\n================================\n%s, t = %d\n================================" % (x,t_idx))
                    for r in rank:
                        text_file.write("\n%-20s %6.4f" % (self.labels[r], cos[r]))

def define_model(args, d, logdir):
    if args.dynamic:
        m = dynamic_bern_emb_model(args, d, logdir)
    else:
        m = bern_emb_model(args, d, logdir)
    return m
