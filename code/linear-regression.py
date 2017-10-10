import numpy as np
import matplotlib.pyplot as plt
import time
import pylab as pl
from matplotlib.animation import FuncAnimation

def data_gen(N=100, mu=0, sigma=.4, xstart=0, xend=1):
    x = np.linspace(xstart,xend,N)
    m, c = .5, 2    
    y = m * x + c + np.random.normal(mu, sigma, N)

    return x,y

def data_plot(x, y):
    plt.figure(1)
    plt.title('y vs. x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0,20,0,12])
    plt.plot(x,y,'.')
    plt.show()

def lr_vector_calculus(x, y):
    X = np.ones([len(x), 2])
    X[:,1] = x
    Y = np.ones([len(y), 1])
    Y[:,0] = y

    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)

    theta = np.dot(np.linalg.inv(XtX), XtY)
    return theta[0], theta[1]

def lr(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    xx = np.multiply(x, x - mean_x)
    yy = np.multiply(x, y - mean_y)

    t1 = np.sum(yy) / np.sum(xx)
    t0 = mean_y - t1 * mean_x

    return t0, t1

def lr_plot(x, y, lr_method):
    t0, t1 = lr_method(x, y)
    y_estimate = t0 + t1*np.linspace(0,20,100)

    plt.figure(2)
    plt.title('Estimated y vs. x')
    plt.text(2, 10, '$\Theta$: (%f, %f)' % (t0, t1))
    plt.axis([0,20,0,12])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y,'b.')
    plt.plot(np.linspace(0,20,100),y_estimate,'r')    
    plt.show()

def gdc_compute_error(t1, t2, x, y):
    N = float(len(x))

    err = 0
    for i in range(0, len(x)):
        err = err + (y[i] - t1 - t2*x[i]) ** 2
    err = err / N    
    return err

def gdc_batch(t1, t2, x, y, learning_rate):
    dt1 = 0
    dt2 = 0
    N = float(len(x))
    for i in range(len(x)):
        dt1 = dt1 - (2/N) * (y[i] - t1 - t2*x[i]) 
        dt2 = dt2 - (2/N) * x[i] * (y[i] - t1 - t2*x[i])
        
    t1 = t1 - (learning_rate * dt1)
    t2 = t2 - (learning_rate * dt2)

    err = gdc_compute_error(t1, t2, x, y)
    
    return t1, t2, err

def gdc_online(t1, t2, x, y, learning_rate):
    N = float(len(x))
    for i in range(len(x)):
        dt1 = -2. * (y[i] - t1 - t2*x[i]) 
        dt2 = -2. * x[i] * (y[i] - t1 - t2*x[i])
        t1 = t1 - (learning_rate * dt1)
        t2 = t2 - (learning_rate * dt2)
    
    err = gdc_compute_error(t1, t2, x, y)
    
    return t1, t2, err

def gdc_run():
    t1, t2 = 10, 10
    num_iterations = 8000
    batch = True

    if batch:
        learning_rate = 0.005
    else:
        learning_rate = 0.001


    errs = []


    for i in range(num_iterations):
        if batch:
            t1, t2, err = gdc_batch(t1, t2, x, y, learning_rate)        
        else:
            t1, t2, err = gdc_online(t1, t2, x, y, learning_rate)

        errs.append(err)
        
        if i % 50 == 0:
            y_estimate = t1 + t2*np.linspace(0,20,100)
                
            plt.figure(3)
            plt.title('Estimated y vs. x')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis([0, 20, 0, 12])
            plt.text(2, 10, '$\Theta$: (%f, %f)' % (t1, t2))
            plt.text(2, 9, 'Err: %f' % err)
            plt.text(2, 8, 'Iteration: %d' % i)
            if batch:
                plt.text(2, 11, 'Batch gradient descent')
            else:
                plt.text(2, 11, 'Online gradient descent')
            plt.plot(x,y,'g.')
            plt.plot(np.linspace(0,20,100),y_estimate,'b')
            plt.pause(0.05)
            plt.clf()
            
    print 't1=', t1, 't2=', t2
    print 'err=', err
    print 'iterations=', i

    # plt.title('Error vs iterations')
    # plt.semilogy(np.linspace(1,len(errs),len(errs)), errs)
    # plt.xlabel('Iterations')
    # plt.ylabel('$\log(err)$')    

def lr_error():
    m, c = .5, 2    
    m_range = np.linspace(m-1115, m+1115, 100)
    c_range = np.linspace(c-1115, c+1115, 100)
    mm, cc = np.meshgrid(m_range, c_range)
    z = np.zeros([len(m_range), len(c_range)])
    for i in range(len(m_range)):
        for j in range(len(c_range)):
            z[i,j] = gdc_compute_error(mm[i,j], cc[i,j], x, y)   

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    jet = plt.get_cmap('jet')
    plt.title('$J(\Theta)$')
    plt.xlabel('$\Theta_1$')
    plt.xlabel('$\Theta_0$')
    plt.axis('equal')
    surf = ax.plot_surface(mm, cc, z, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)
    ax.set_zlim3d(0, z.max())
    plt.show()    

def datasets_autompg():
    import numpy
    usecols = {0,1,2,3, 4,5,6}
    converters = {3: lambda s: float(s.strip() != '?' or 0)}
    auto_mpg = numpy.loadtxt('../../datasets/regression/auto-mpg/auto-mpg.data', converters=converters, usecols=usecols)    

    x = auto_mpg[:,2]
    y = auto_mpg[:,0]

    t0, t1 = lr(x, y)

    plt.title('auto-mpg')
    plt.ylabel('mpg')
    plt.xlabel('displacement')
    plt.plot(x,y,'.')
    y_estimate1 = t0 + t1*np.linspace(np.min(x)-10,np.max(x)+10,10)
    plt.text(2, 10, '$\Theta$: (%f, %f)' % (t0, t1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y,'b.')
    plt.plot(np.linspace(np.min(x)-10,np.max(x)+10,10),y_estimate1,'r')
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)

    x, y = data_gen(xstart=5, xend=15)    
    #data_plot(x, y)

    #lr_plot(x, y, lr)
    #lr_plot(x, y, lr_vector_calculus)

    #gdc_run()

    #lr_error()

    datasets_autompg()

    # # fig = plt.gcf()
    # # fig.show()  
    # # fig.canvas.draw()

    # for i in range(4):
    #     plt.figure(1)
    #     x, y = data_gen()
    #     plt.plot(x,y,'.')
    #     # fig.canvas.draw()
    #     plt.pause(1.0)
    #     plt.clf()
