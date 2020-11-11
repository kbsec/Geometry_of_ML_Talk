import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


def synthetic_example(mu=0, sigma = 1,N=400, c_boundary=False,y_ints=[5,6,7], SEED=4):
    np.random.seed(SEED)
    plt.figure(figsize=(9,9))
    plt.title("Synthetic Data Example", fontsize=20)
    c1 =  np.ones( (2,N)) + np.random.normal(0,sigma,(2,N))
    c2 =  5 + np.zeros( (2,N)) + np.random.normal(0,sigma,(2,N))
    plt.scatter(c1[0], c1[1], edgecolors='b', label='Malignant Tumor')
    plt.scatter(c2[0], c2[1], c='r', edgecolors='b', label='Benign  Tumor')
    if c_boundary:
        xb = [i for i in range(-2,9)]
        dc=1
        for j in y_ints:
            yb = [-1 * i + j for i in xb]
            plt.plot(xb,yb,label='Desicion Boundary '+ str(dc) ,linewidth=2 )
            dc += 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. , fontsize=18)
    
    plt.grid(True)
    plt.xlabel("Feature 1", fontsize=18)
    
    plt.ylabel("Feature 2", fontsize=18)
    labels1 =  np.zeros(N)
    labels2 = np.ones(N)
    y =  np.concatenate((labels1,labels2),axis=0)
    x0 =  np.concatenate((c1[0],c2[0]),axis=0)
    x1 =  np.concatenate((c1[1],c2[1]),axis=0)
    X=np.array([x0,x1,y]).T
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Target'])
    plt.show()
    return df



def log_likelihood(X, p):
    """
    Returns the log likelhood of observing X given Pr(X_i=p)
    """
    pos_prob  = X * np.log(p)
    neg_prob = (1-X) * np.log(1-p)
    return np.sum(pos_prob + neg_prob)
def heads_tails(H, T):
    heads = np.ones(H)
    tails = np.zeros(T)
    return np.concatenate( (heads, tails))


def cost_values(H,T, p_grid):
    X = heads_tails(H,T)
    costs = [log_likelihood(X,p) for p in p_grid]
    return costs 

def plot_cost_graph(H,T):
    X = heads_tails(H,T)
    plt.figure(figsize=(8,8))
    print("PLotting")
    p_grid = np.arange(.01, 1,.01)
    costs = [log_likelihood(X,p) for p in p_grid]
    plt.plot(p_grid, costs , c='r')
    opt = np.sum(X)/ len(X)
    cost_opt = log_likelihood(X, opt)
    plt.scatter(opt, cost_opt, marker='x', s=100, c='pink' )
    plt.title("Maximum Likelihood Solution")
    plt.xlabel("Probability of Heads $p$")
    plt.ylabel("Log Likelihood")
    plt.ylim(-250, -50)
    plt.xlim(0,1)
    plt.show()



def g(z):
    """
    This function computes the sigmoid function across all values of z

    Argument:
    z -- numpy array of real numbers

    Returns:
    sigmoid(z)
    """
    
    return 1 / (1 + np.exp(-z))

def h(b, w ,X):
    """
    This function implments the logistic regression hypothesis function

    Argument:
    b -- bias
    w -- predictive parameters
    X -- data matrix of size (numbers_examples, number_predictors)

    Returns:
    sigmoid(Xw + b)
    """
    return g( (X @ w) + b)

def computeCost(b, w, X, Y): 
    """
    Computes Cross Entropy Loss function 

    Arguments:
    b -- bias
    w -- predictive parameters
    X -- data matrix of size (numbers_examples, number_predictors)
    Y -- Ground truth labels of size (number_examples, 1)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    """
    m = len(Y.flatten())
    assert m >0
    term1 = np.dot(-np.array(Y).T,np.log( np.maximum(h(b,w,X), 1e-9)))
    term2 = np.dot((1-np.array(Y)).T,np.log( np.maximum(1-h(b,w,X) , 1e-9) ))
    return float( (1./m) * ( np.sum(term1 - term2) ) )


from IPython.core.display import display, HTML
import json

def plot3D(X, Y, Z, height=600, xlabel = "X", ylabel = "Y", zlabel = "Cost", initialCamera = None, save_fie=False):

    options = {
        "width": "100%",
        "style": "surface",
        "showPerspective": True,
        "showGrid": True,
        "showShadow": False,
        "keepAspectRatio": True,
        "height": str(height) + "px"
    }

    if initialCamera:
        options["cameraPosition"] = initialCamera

    data = [ {"x": X[y,x], "y": Y[y,x], "z": Z[y,x]} for y in range(X.shape[0]) for x in range(X.shape[1]) ]
    visCode = r"""
       <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" type="text/css" rel="stylesheet" />
       <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
       <div id="pos" style="top:0px;left:0px;position:absolute;"></div>
       <div id="visualization"></div>
       <script type="text/javascript">
        var data = new vis.DataSet();
        data.add(""" + json.dumps(data) + """);
        var options = """ + json.dumps(options) + """;
        var container = document.getElementById("visualization");
        var graph3d = new vis.Graph3d(container, data, options);
        graph3d.on("cameraPositionChange", function(evt)
        {
            elem = document.getElementById("pos");
            elem.innerHTML = "H: " + evt.horizontal + "<br>V: " + evt.vertical + "<br>D: " + evt.distance;
        });
       </script>
    """
    htmlCode = "<iframe srcdoc='"+visCode+"' width='100%' height='" + str(height) + "px' style='border:0;' scrolling='no'> </iframe>"
    if save_fie:
        Html_file= open(save_file,"w")
        Html_file.write(htmlCode)
        Html_file.close()
        
    display(HTML(htmlCode))
    

    def plot_logistic(X,y, tr=False):
        X_ = X
        if tr:
            x3 = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2))
            x_ = np.hstack((X, x3))
        clf = LogisticRegression().fit(X_, y)
        xx, yy = np.mgrid[-2:2:.01, -2:2:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
        f , ax = plt.subplots(figsize=(15, 15))
        
        contour = ax.contourf(xx, yy, probs, 30, cmap="RdBu",
                            vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(X[:,0], X[:, 1], c=y, s=50,
                cmap="RdBu", vmin=-.2, vmax=1.2,
                edgecolor="white", linewidth=1)

        ax.set(aspect="equal",
            xlim=(-2, 2), ylim=(-2, 2),
            xlabel="$X_1$", ylabel="$X_2$")
        plt.show()
        
def Gaussian_plot():
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 1.])
    Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=(16,16))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap='viridis')

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap='viridis')

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)

    plt.show()
    return X,Y,Z



def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.3 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X, Y




def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))



from itertools import chain, product
from bqplot import *
class NeuralNet(Figure):
    def __init__(self, **kwargs):
        self.height = kwargs.get('height', 600)
        self.width = kwargs.get('width', 960)
        self.directed_links = kwargs.get('directed_links', False)
        
        self.num_inputs = kwargs['num_inputs']
        self.num_hidden_layers = kwargs['num_hidden_layers']
        # add in the weight vectors 
        self.nodes_output_layer = kwargs['num_outputs']
        self.layer_colors = kwargs.get('layer_colors', 
                                       ['Orange'] * (len(self.num_hidden_layers) + 2))
        
        self.build_net()
        super(NeuralNet, self).__init__(**kwargs)
    
    def build_net(self):
        # create nodes
        self.layer_nodes = []
        self.layer_nodes.append(['x' + str(i+1) for i in range(self.num_inputs)])
        
        for i, h in enumerate(self.num_hidden_layers):
            self.layer_nodes.append(['h' + str(i+1) + ',' + str(j+1) for j in range(h)])
        self.layer_nodes.append(['y' + str(i+1) for i in range(self.nodes_output_layer)])
        
        self.flattened_layer_nodes = list(chain(*self.layer_nodes))
        
        # build link matrix
        i = 0
        node_indices = {}
        for layer in self.layer_nodes:
            for node in layer:
                node_indices[node] = i
                i += 1

        n = len(self.flattened_layer_nodes)
        self.link_matrix = np.empty((n,n))
        self.link_matrix[:] = np.nan

        for i in range(len(self.layer_nodes) - 1):
            curr_layer_nodes_indices = [node_indices[d] for d in self.layer_nodes[i]]
            next_layer_nodes = [node_indices[d] for d in self.layer_nodes[i+1]]
            for s, t in product(curr_layer_nodes_indices, next_layer_nodes):
                self.link_matrix[s, t] = 1
        
        # set node x locations
        self.nodes_x = np.repeat(np.linspace(0, 100, 
                                             len(self.layer_nodes) + 1, 
                                             endpoint=False)[1:], 
                                 [len(n) for n in self.layer_nodes])

        # set node y locations
        self.nodes_y = np.array([])
        for layer in self.layer_nodes:
            n = len(layer)
            ys = np.linspace(0, 100, n+1, endpoint=False)[1:]
            self.nodes_y = np.append(self.nodes_y, ys[::-1])
        
        # set node colors
        n_layers = len(self.layer_nodes)
        self.node_colors = np.repeat(np.array(self.layer_colors[:n_layers]), 
                                     [len(layer) for layer in self.layer_nodes]).tolist()
        
        xs = LinearScale(min=0, max=100)
        ys = LinearScale(min=0, max=100)
        
        self.graph = Graph(node_data=[{'label': d, 
                                       'label_display': 'none'} for d in self.flattened_layer_nodes], 
                           link_matrix=self.link_matrix, 
                           link_type='line',
                           colors=self.node_colors,
                           directed=self.directed_links,
                           scales={'x': xs, 'y': ys}, 
                           x=self.nodes_x, 
                           y=self.nodes_y,
                           # color=2 * np.random.rand(len(self.flattened_layer_nodes)) - 1
                          )
        self.graph.hovered_style = {'stroke': '1.5'}
        self.graph.unhovered_style = {'opacity': '0.4'}
        
        self.graph.selected_style = {'opacity': '1',
                                     'stroke': 'red',
                                     'stroke-width': '2.5'}
        self.marks = [self.graph]
        self.title = 'Neural Network'
        self.layout.width = str(self.width) + 'px'
        self.layout.height = str(self.height) + 'px'