import time
import numpy as np
import pickle
from numpy.random import seed
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from utils import *


#dataset = data_preproc(np.asarray(pickle.load(open('dataset/EXP1.pickle','rb'))))

def exp1():
    generator = load_model('models/GAN/exp1/GAN-g-exp1-16.h5')
    classifier = load_model('models/GAN/exp1/GAN-c-exp1-6400.h5')
    # use GAN to produce random samples
    latent_dim = 100
    z_inputs = generate_latent_points(latent_dim, 6400)
    generated_samples = generator.predict(z_inputs)

    y_g = classifier.predict(generated_samples)
    y_g = np.argmax(y_g, axis=-1)
    generated_samples = generated_samples.reshape(-1,120)

    print(y_g.shape)
    print(generated_samples.shape)
    generated_samples = single_minmaxscale(generated_samples, scale_range=(0,1))

    for target in range(16):
        plt.figure()
        for x_g in generated_samples[y_g==target]:
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42        
            plt.title('Location $p_{%d}$ (fake)'%(target+1) , fontsize=20)
            plt.xlabel('CSI Index', fontsize=18)
            plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
            plt.axis([0, 120, 0, 1])
            plt.grid(True)

            plt.plot(x_g)

        plt.savefig('visualizations/GAN/EXP1/GAN-exp1-p%d.png'%(target+1), dpi=400)
        plt.savefig('visualizations/GAN/EXP1/GAN-exp1-p%d.eps'%(target+1), dpi=1)  
        plt.close() 


def exp2():
    generator = load_model('models/GAN/exp2/GAN-g-exp2-14.h5')
    classifier = load_model('models/GAN/exp2/GAN-c-exp2-10752.h5')
    # use GAN to produce random samples
    latent_dim = 100
    z_inputs = generate_latent_points(latent_dim, 6400)
    generated_samples = generator.predict(z_inputs)

    y_g = classifier.predict(generated_samples)
    y_g = np.argmax(y_g, axis=-1)
    generated_samples = generated_samples.reshape(-1,120)

    print(y_g.shape)
    print(generated_samples.shape)
    generated_samples = single_minmaxscale(generated_samples, scale_range=(0,1))

    for target in range(14):
        for x_g in generated_samples[y_g==target]:
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42            
            plt.title('Location $p_{%d}$ (fake)'%(target+1) , fontsize=20)
            plt.xlabel('CSI Index', fontsize=18)
            plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
            plt.axis([0, 120, 0, 1])
            plt.grid(True)

            plt.plot(x_g)
        plt.savefig('visualizations/GAN/EXP2/GAN-exp2-p%d.png'%(target+1), dpi=400)
        plt.savefig('visualizations/GAN/EXP2/GAN-exp2-p%d.eps'%(target+1), dpi=1)  
        plt.close() 

def exp3():
    generator = load_model('models/GAN/exp3/GAN-g-exp3-18.h5')
    classifier = load_model('models/GAN/exp3/GAN-c-exp3-18.h5')
    # use GAN to produce random samples
    latent_dim = 100
    z_inputs = generate_latent_points(latent_dim, 6400)
    generated_samples = generator.predict(z_inputs)

    y_g = classifier.predict(generated_samples)
    y_g = np.argmax(y_g, axis=-1)
    generated_samples = generated_samples.reshape(-1,120)

    print(y_g.shape)
    print(generated_samples.shape)
    generated_samples = single_minmaxscale(generated_samples, scale_range=(0,1))
    

    for target in range(18):
        plt.figure()
        for x_g in generated_samples[y_g==target]:
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42            
            plt.title('Location $p_{%d}$ (fake)'%(target+1) , fontsize=20)
            plt.xlabel('CSI Index', fontsize=18)
            plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
            plt.axis([0, 120, 0, 1])
            plt.grid(True)

            plt.plot(x_g)

        plt.savefig('visualizations/GAN/EXP3/GAN-exp3-p%d.png'%(target+1), dpi=400)
        plt.savefig('visualizations/GAN/EXP3/GAN-exp3-p%d.eps'%(target+1), dpi=1)  
        plt.close() 

def test():
    generator = load_model('models/GAN/exp1/GAN-g-exp1-16.h5')
    classifier = load_model('models/GAN/exp1/GAN-c-exp1-6400.h5')
    # use GAN to produce random samples
    latent_dim = 100
    z_inputs = generate_latent_points(latent_dim, 6400)
    generated_samples = generator.predict(z_inputs)

    y_g = classifier.predict(generated_samples)
    y_g = np.argmax(y_g, axis=-1)
    generated_samples = generated_samples.reshape(-1,120)

    for target in range(16):
        plt.figure()
        for x_g in generated_samples[y_g==target]:
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['ps.fonttype'] = 42        
            plt.title('Location $p_{%d}$ (fake)'%(target+1) , fontsize=20)
            plt.xlabel('CSI Index', fontsize=18)
            plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
        
            plt.grid(True)

            plt.plot(x_g)

        #plt.savefig('visualizations/GAN/EXP1/GAN-exp1-p%d.png'%(target+1), dpi=400)
        #plt.savefig('visualizations/GAN/EXP1/GAN-exp1-p%d.eps'%(target+1), dpi=1)  
        #plt.close()
        plt.show() 

if __name__ == '__main__':

    seed(0)
    exp1()
    exp2()
    exp3()
