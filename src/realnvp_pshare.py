import torch
import numpy as np

class Rescale(torch.nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1, num_channels))

    def forward(self, x):
        x = self.weight * x
        return x

class RealNVP(torch.nn.Module):
    """ This is a class for implementing the Real NVP (Non-Volume Preserving) network 
    for Density Estimation. It is based on the principle of invertible, bijective mappings
    or transformations that map a given complex distribution of data to a latent space, 
    through tractable density and sampling calculation as well as tractable inverses.
    The main function is the change of variable rule that is defined by :

    P_x = P_z * det ( df(x) / dx )

    Normalizing (Inference): z = f(x)
    Generation (Sampling): x = g(z)
    -------
    Args: 
          nets: Defines a Network that is to be used as the 'conditioner' function.
                Also serves for implementation of the scale (s) and the transition(t)
                functions of the network

          mask: The transformations are defined often through the use of masks which are
                applied as element wise operations on the input. Mostly they help to split 
                the data into two halves (the identical half and the scaling half) at each 
                coupling layer

          prior: This defines the prior to be used for the distribution on the latent space.
                 Usually, the prior is modeled using a Standard multivariate normal distribution
                 (with zero mean vector  and identity covariance matrix)
    
    Methods: 
          _chunk() : This function is used to divide the function into two possible halves. The 
                  division is decided by the mask applied and takes place along the second dimension.
                
          g() : This function is actually the sampling function that is used for generating a point 
             in the data space X from a point in the latent space Z. 
            
          f() : This function is actually the inference function that is used for inferring a point 
             in the latent space Z from a point in the data space X. It also additionally
             calculates the log of the determinant of the Jacobian matrix that is associated
             with the RealNVP transformation.

          log_prob() : This function is used for estimating the logarithm of the prob density of 
                       the data points in the Data Space X. It follows the change of variable formula
                       as outlined above.
          
          sample() : This function is used to draw a sample from the learnt distribution of the data
                     from the latent space Z
             
    """
    """
    RealNVP module.
    Adapted from https://github.com/senya-ashukha/real-nvp-pytorch
    """
    def __init__(self, mask, prior, tied_states=False, st_nets=None, t_nets=None):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        if tied_states is not True:
            self.s = torch.nn.ModuleList([st_nets() for _ in range(len(mask))])
            self.rescale = torch.nn.utils.weight_norm(Rescale(int(self.mask.size(1)/2)))
        else:
            self.t = torch.nn.ModuleList([t_nets() for _ in range(len(mask))])
            self.rescale = torch.nn.utils.weight_norm(Rescale(int(self.mask.size(1)/2)))
    
    
    def _chunk(self, x, mask):
        """chunk the input x into two chunks along dimension 2
        INPUT: tensor to be chunked, shape: batch_size * n_samples * n_features
        OUTPUT: tow chunks of tensors with equal size
        """
        idx_id = torch.nonzero(mask).reshape(-1)
        idx_scale = (mask == 0).nonzero().reshape(-1)
        #idx_scale = torch.nonzero(~mask).reshape(-1)
        chunk_id = torch.index_select(x, dim=2,
                                      index=idx_id)
        chunk_scale = torch.index_select(x, dim=2,
                                         index=idx_scale)
        return (chunk_id, chunk_scale)
        
    def g(self, z):
        """ This function defines the generation or sampling step. It is the exact opposite of the inference
        step and the masking operations are similar. 

        """
        # not sure about this
        x = z
        # for i in range(len(self.t)):
        #     x_ = x * self.mask[i]
        #     s = self.s[i](x_) * (1 - self.mask[i])
        #     t = self.t[i](x_) * (1 - self.mask[i])
        #     x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        # return x
        pass

    def f(self, x, s_shared = None):
        """ This function defines the inference step. The flow of the network is something like:
        
        Z (Latent Space) <-- f^{1} <-- f^{2} ... <-- f^{L} <-- X (Data Space)
        
        Hence, the list is reversed first and then iterated over all the layers
        The input is split into two parts z_id and z_s. The part involving z_id is 
        passed through as identical and the part involving z_id is subjected to two operations of 
        scaling (s(z_id)) and translation (t(z_id)) and mxied with z_s. 
        
        The scaling and the translation output obtained here
        as variables s and t are chunked. The scaling output is actually to processed as tanh and the
        translation output remains as an affine transform output. 
        
        The determinant of the jacobian matrix of this computation step is given as a sum of the
        scaling outputs and thus easily computed without derivatives of s or t functions
        """
        self.s_shared = s_shared
        log_det_J, z = x.new_zeros((x.shape[0], x.shape[1])), x
        for i in reversed(range(len(self.s))):
            
            if self.s_shared == None:
                z_id, z_s = self._chunk(z, self.mask[i])
                st = self.s[i](z_id)
                s, t = st.chunk(2,dim=2)
                s = self.rescale(torch.tanh(s))
                exp_s = s.exp()
                z_s = (z_s + t) * exp_s
                z =  torch.cat((z_id, z_s), dim=2)
                log_det_J += torch.sum(s, dim=2)
            else:
                z_id, z_s = self._chunk(z, self.mask[i])
                #st = self.s_shared[i](z_id)
                #s, t = st.chunk(2,dim=2)
                t = self.t[i](z_id)
                s = self.s_shared[i](z_id)
                s = self.rescale(torch.tanh(s))
                exp_s = s.exp()
                z_s = (z_s + t) * exp_s
                z =  torch.cat((z_id, z_s), dim=2)
                log_det_J += torch.sum(s, dim=2)
        
        return z, log_det_J

    def log_prob(self, x, mask, s_shared=None):
        """The prior log_prob may need be implemented such it adapts cuda computation."""
        z, logp = self.f(x, s_shared)

        px = self.prior.log_prob(z) + logp
        # set the padded positions as zeros
        px[~mask] = 0
        # px[~mask].zero_()
        #if (px > 0).any():
          #  print("here")
        return px

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

class RealNVP_FlowNets(torch.nn.Module):
    """Create an array of flow networks of shape(n_states, n_components).
    The scaling networks are to be defined for per state basis if tied-states
    are used. Else, if tied-states are used, the scaling networks are defined 
    for every network
    ----
    Args:
        - n_states: No. of states in HMM
        - n_prob_components: No. of mixture components
    Returns:
        - self
    """
    def __init__(self, n_states, n_prob_components, net_s, net_t, net_st, masks, prior, device, use_tied_states=False):
        super(RealNVP_FlowNets, self).__init__()
        
        self.n_states = n_states # Number of states
        self.n_prob_components = n_prob_components # Number of mixture components
        #self.s_nets = torch.nn.ModuleList([net_s() for _ in range(len(mask))]) # scaling networks (in case one network is used for every state and every mixture component)
        # Creating a list of s_nets for every state, which is to be shared between the mixture components per state
        self.s_nets_per_state = [torch.nn.ModuleList([net_s() for _ in range(len(masks))]) for _ in range(self.n_states)]
        self.t_nets = torch.nn.ModuleList([net_t() for _ in range(len(masks))]) # translational networks
        self.device = device
        self.masks = masks
        self.prior = prior
        self.use_tied_states = use_tied_states # flag to denote whether to use tied states or not
        #self.init_mixture() # Initialise the mixture weights

        if use_tied_states == True:

            # Initialise the networks for each component and state
            self.networks = [RealNVP(self.masks, self.prior, tied_states=self.use_tied_states, st_nets=st_nets, t_nets=t_nets)
                         for _ in range(self.n_prob_components*self.n_states)]

            # Reshape the new networks in a n_states x n_prob_components array
            self.networks = np.array(self.networks).reshape(
                self.n_states, self.n_prob_components)
        
        else:
            
            # Initialise the networks for each component and state
            self.networks = [RealNVP(self.masks, self.prior, tied_states=False, st_nets=net_st)
                         for _ in range(self.n_prob_components*self.n_states)]

            # Reshape the new networks in a n_states x n_prob_components array
            self.networks = np.array(self.networks).reshape(
                self.n_states, self.n_prob_components)

    def _getllh(self, logPIk_s, batch):
        """This function is used to obtain the log-likelihood of the given batch of data
        ----
        Args:
            batch ([tuple of tensors]): Contains an Input tensor of the shape (batch_size, n_samples, n_features)
                                        and a tensor representing the mask for the input tensor that indicates the 
                                        actual number of input samples present
        """
        x, x_mask = batch 
        batch_size = x.shape[0] # Obtain the batch size
        n_samples = x.shape[1] # Obtain the actual number of samples

        # Initialise an output tensor that stores the log-likelihood p(x|s) for each state s
        llh = torch.zeros((batch_size, n_samples, self.n_states))

        # Initialise an output tensor that stores the log-likelihood p(x|s,k) for each mixture component k and each state s
        local_llh_sk = torch.zeros((batch_size, n_samples, self.n_states, self.n_prob_components))

        for s in range(self.n_states):
            
            if self.use_tied_states == True:
                
                # Compute the log-prob for each component k and each state s 
                # Except that here each mixture component uses the same scaling network
                llh_sk = [self.networks[s,k].log_prob(x, x_mask, s_shared=self.s_nets_per_state[s]) / x.size(2) 
                            for k in range(self.n_prob_components)]

                # torch.stack() concatenates the seqeunce of llh_sk tensors along a new dimension (which is by default 0)
                # hence, when the resulting tensor is added it has be permuted so that the resulting shape
                # is (batch_size, n_samples, ...)
                ll = torch.stack(llh_sk).permute(1, 2, 0)

                # Assign the resulting value to the collecting tensor local_llh_sk
                local_llh_sk[:, :, s, :] = ll

                # Implements the equation using the logsumexp trick:
                # - log p(x|s) = log \sum_{k=1}^{K_g} \pi_{s,k} p(x|s,k)
                # The logsumexp trick is applied along the dimension corresponding to the n_prob_components
                llh[:, :, s] = torch.logsumexp((self.logPIk_s[s].reshape(1, 1, self.n_prob_components) + ll).detach(), dim=2)
            
            else:

                # Compute the log-prob for each component k and each state s
                llh_sk = [self.networks[s,k].log_prob(x, x_mask)/x.size(2) for k in range(self.n_prob_components)]
                
                # torch.stack() concatenates the seqeunce of llh_sk tensors along a new dimension (which is by default 0)
                # hence, when the resulting tensor is added it has be permuted so that the resulting shape
                # is (batch_size, n_samples, ...)
                ll = torch.stack(llh_sk).permute(1, 2, 0)

                # Assign the resulting value to the collecting tensor local_llh_sk
                local_llh_sk[:, :, s, :] = ll

                # Implements the equation using the logsumexp trick
                # log p(x|s) = log \sum_{k=1}^{K_g} \pi_{s,k} p(x|s,k)
                # NOTE: The logsumexp trick is applied along the dimension corresponding to the
                # n_prob_components
                llh[:, :, s] = torch.logsumexp((logPIk_s[s].reshape(1, 1, self.n_prob_components) + ll).detach(), dim=2)

        return llh, local_llh_sk
