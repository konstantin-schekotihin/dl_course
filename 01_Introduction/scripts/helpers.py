import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

plt_x, plt_y = 8, 8

def plot_out(out):
    print("Finished optimization in {} iterations with the grad {}".format(len(out),out[-1]))
    plt.plot(out)
    plt.yscale('log')
    plt.show()
    
def annotate(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}, p = {:.3f} ".format(r, p),
                xy=(.1, 1), xycoords=ax.transAxes)

def plot_residuals(predict, X, y):
    est = predict(X.reshape(-1,1)).reshape(-1,)
    rss = torch.sum(torch.pow(y-est, 2))
    tss = torch.sum(torch.pow(y-torch.mean(y), 2))
    print("TSS = {:.3f} - total sum of squares - squared error of an average predictor".format(tss))
    print("RSS = {:.3f} - residual sum of squares".format(rss))
    print("ESS = TSS - RSS =  {:.3f} - {:.3f} = {:.3f} - explained sum of squares".format(tss, rss, tss - rss))
    print("MSE = {:.3f} - mean squared error".format(rss/est.shape[0]))
    print("MAE = {:.3f} - mean absolute error".format(torch.sum(torch.abs(y-est))/est.shape[0]))
    l = torch.linspace(0, 300, 1000).reshape(-1,1)
    
    plt.figure(figsize=(plt_x, plt_y))
    plt.plot([X.numpy(), X.numpy()], [y.numpy(), est.numpy()], 'bo--')
    plt.plot(X, est, 'ro')
    plt.plot(l, predict(l.float()), 'g')
    plt.show();

def g_idea(f, df, x=1, h=0.05):
    sp = np.linspace(-.5,1,100)
    plt.plot(sp, f(sp))
    x_n = x-h*df(x)
    plt.arrow(x, f(x), x_n-x, f(x_n)-f(x), head_width=0.04,length_includes_head=True, color='r', linewidth=3)
    plt.plot(x, f(x), 'ro')
    plt.show()

def plot_trace(f, trace):
    n = max(abs(min(trace)), abs(max(trace)))
    domain = torch.arange(-n, n, 0.01)
    plt.plot(domain, f(domain))
    plt.plot(trace, [f(x) for x in trace], '-ro')
    plt.show()

def plot_grad(f,x,y,x1,y1,angle):
    fig = plt.figure(figsize=(plt_x, plt_y))
       
    sp = np.linspace(-10,10,1000)
    X, Y = np.meshgrid(sp,sp)
    Z = f(X,Y)

    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, angle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')

    ax.plot_surface(X, Y, Z, alpha=0.3, cmap=plt.cm.summer, linewidth=0.1)
    ax.scatter([x,x],[y,y],[0,f(x,y)], s=40, c='r')
    ax.scatter([x1,x1],[y1,y1],[f(x1,y1),0],  s=40, c='b')
    ax.plot([x,x1],[y,y1],[f(x,y),f(x1,y1)], color='black')
    ax.plot([x,x1],[y,y1],[0,0], color='black')
    print("initial: ", [x,y,f(x,y)], " step: ", [x1, y1, round(f(x1,y1),2)])
    plt.show()
    
def run_opt(opt, x=-8, y=0, h = 0.1, angle=60):
    opt.step(x, y, h)
    opt.plot(x, y, angle)
    opt.update_widgets()
    
def run_momentum(opt, x=-8, y=0, h = 0.1, mu = 0.5, angle=60):
    opt.set_mu(mu)
    run_opt(opt, x, y, h, angle)