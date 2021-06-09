import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from numpy.random import seed

from models import *
from utils import *

def fit_GAN(run, g_model, d_model, c_model, gan_model, n_samples, n_classes, X_sup, y_sup, dataset, n_epochs, n_batch, latent_dim = 100):
    tst_history = []
    X_tra, y_tra, X_tst, y_tst = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(X_tra.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # fit the model
    for i in range(n_steps):
        # update discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update discriminator (d)
        [X_real, _], y_real = generate_real_samples((X_tra, y_tra), half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d/%d/%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (run+1, i+1, n_steps, c_loss, c_acc*100, d_loss1[0], d_loss2[0], g_loss))
        # test after a epoch
        if (i+1) % (bat_per_epo * 1) == 0:
            _, _acc = c_model.evaluate(X_tst, y_tst, verbose=0)
            tst_history.append(_acc)

    return tst_history

def select_supervised_samples(X, Y, n_samples, n_classes):
    X_list, Y_list = list(), list()
    n_per_class = int(n_samples/n_classes)

    for i in range(n_classes):
        X_with_class = X[Y==i]
        ix = np.random.randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [Y_list.append(i) for j in ix]
    return np.asarray(X_list), np.asarray(Y_list)

def generate_real_samples(dataset, n_samples):

    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    Y=np.ones((n_samples, 1))
    return [X, labels], Y 

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)

    return z_input  

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict(z_input)
    y = np.zeros((n_samples, 1))
    return images, y   

def run_exp1():
    #experiment setup
    n_classes = 16
    n_samples = [16] # define the number of labeled samples here
    run_times = 1 # define the number of runs to traing under this setting
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    n_epochs = 100
    n_batch = 128

    #load dataset
    dataset = data_preproc(np.asarray(pickle.load(open('dataset/EXP1.pickle','rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j],  n_classes)        
        for i in range(run_times):
            print('{}/{}'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
            #history.append(tst_acc)
        best = max (history)
        # save results:
        fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        pickle.dump(history, fh)
        fh.close()
        # save models:
        #g_model.save('exp1_result/GAN-g-exp1-{}-{}.h5'.format(n_samples[j], int(best*100)))
        #d_model.save('exp1_result/GAN-d-exp1-{}-{}.h5'.format(n_samples[j], int(best*100)))
        #c_model.save('exp1_result/GAN-c-exp1-{}-{}.h5'.format(n_samples[j], int(best*100)))


def run_exp2():
    #experiment setup
    n_classes = 14
    n_samples = [14] # define the number of labeled samples here
    run_times = 1
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    n_epochs = 100
    n_batch = 128

    #load dataset
    dataset = data_preproc(np.asarray(pickle.load(open('dataset/EXP2.pickle','rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup, y_sup = select_supervised_samples(X_tra, y_tra, n_samples[j],  n_classes)        
        for i in range(run_times):
            print('{}/{}'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
            #history.append(tst_acc)
        best = max (history)
        # save results:
        #fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #fh = open('GAN-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #pickle.dump(history, fh)
        #fh.close()
        # save models:
        #g_model.save('exp2_result/GAN-g-exp2-{}-{}.h5'.format(n_samples[j], int(best*100)))
        #d_model.save('exp2_result/GAN-d-exp2-{}-{}.h5'.format(n_samples[j], int(best*100)))
        #c_model.save('exp2_result/GAN-c-exp2-{}-{}.h5'.format(n_samples[j], int(best*100)))




def run_exp3():
    # experiment setup #Train: 400/loc, 6400 in total
    n_classes = 18
    n_samples = [3600]
    run_times = 1
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    n_epochs = 100
    n_batch = 128

    #load dataset
    dataset1 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r1.pickle','rb'))))
    dataset2 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r2.pickle','rb'))))
    X_tra1, y_tra1, X_tst1, y_tst1 = dataset1
    X_tra2, y_tra2, X_tst2, y_tst2 = dataset2

    # combine the data from r1/r2
    X_tra = np.concatenate((X_tra1, X_tra2))
    y_tra = np.concatenate((y_tra1, y_tra2))
    X_tst = np.concatenate((X_tst1, X_tst2))
    y_tst = np.concatenate((y_tst1, y_tst2))
    dataset = (X_tra, y_tra, X_tst, y_tst)

    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup1, y_sup1 = select_supervised_samples(X_tra1, y_tra1, n_samples[j],  n_classes)
        X_sup2, y_sup2 = select_supervised_samples(X_tra2, y_tra2, n_samples[j],  n_classes)
        X_sup = np.concatenate((X_sup1, X_sup2))
        y_sup = np.concatenate((y_sup1, y_sup2))        
        for i in range(run_times):
            print('{}/{}'.format(i+1, run_times))
            # change seed for each run
            seed(run_times)
            # define a semi-GAN model
            d_model, c_model = define_discriminator(n_classes, optimizer)
            g_model = define_generator()
            gan_model = define_GAN(g_model, d_model, optimizer)

            # train the semi-GAN model
            tst_acc = fit_GAN(i ,g_model, d_model, c_model, gan_model, n_samples[j], n_classes, X_sup, y_sup, dataset, n_epochs, n_batch)

            history.append(max(tst_acc))
            #history.append(tst_acc)
        best = max (history)


        #g_model.save('exp3_result/GAN-g-exp1-{}-{}.h5'.format(n_samples[j], int(best*100)))
        #d_model.save('exp3_result/GAN-d-exp1-{}-{}.h5'.format(n_samples[j], int(best*100)))
        #c_model.save('exp3_result/GAN-c-exp1-{}-{}.h5'.format(n_samples[j], int(best*100)))

        #fh = open('GAN-r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #fh = open('GANr1r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        #pickle.dump(history, fh)
        #fh.close()

def run_cnn():
    '''
    Run CNN under different number of supervised samples
    '''

    # experiment setup
    n_classes = 18
    n_samples = [18, 36, 72, 1800, 3600]
    run_times = 10
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    batch_size = 18 
    epochs = 50

    #load dataset
    dataset1 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r1.pickle','rb'))))
    dataset2 = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r2.pickle','rb'))))
    X_tra1, y_tra1, X_tst1, y_tst1 = dataset1
    X_tra2, y_tra2, X_tst2, y_tst2 = dataset2 
    X_tra = np.concatenate((X_tra1, X_tra2))
    y_tra = np.concatenate((y_tra1, y_tra2))
    X_tst = np.concatenate((X_tst1, X_tst2))
    y_tst = np.concatenate((y_tst1, y_tst2))
    
    for j in range(len(n_samples)):
        history = []
        # select supervised dataset
        X_sup1, y_sup1 = select_supervised_samples(X_tra1, y_tra1, n_samples[j],  n_classes)
        X_sup2, y_sup2 = select_supervised_samples(X_tra2, y_tra2, n_samples[j],  n_classes)
        X_sup = np.concatenate((X_sup1, X_sup2))
        y_sup = np.concatenate((y_sup1, y_sup2))        

        for i in range(run_times):
            seed(run_times)
            model = CNN (n_classes, optimizer)
            print('{}/{}'.format(i+1, run_times))
            model.fit(X_sup, y_sup, batch_size, epochs, verbose = 1)
            tst_acc = model.evaluate(X_tst, y_tst)[1]
            print("Test Acc = {}".format(tst_acc))
            history.append(tst_acc)
        best = max (history)

        #fh = open('GAN-r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        fh = open('CNNr1r2-{}-{}.pickle'.format(n_samples[j], best),'wb')
        pickle.dump(history, fh)
        fh.close()    

if __name__ == '__main__':

    run_exp1()
