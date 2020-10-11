import os
import sys
import numpy as np
from gm_hmm.src.realnvp_pshare import RealNVP, RealNVP_FlowNets
import torch
from torch import nn, distributions
from gm_hmm.src._torch_hmmc import _compute_log_xi_sum, _forward, _backward
from gm_hmm.src.utils import step_learning_rate_decay
from hmmlearn.base import ConvergenceMonitor
from timeit import default_timer as timer

class ConvgMonitor(ConvergenceMonitor):
    def report(self, logprob):
        """
        Reports the convergence to the :data:'sys.stderr'
        (which is the log file for class training hopefully)

        The output consists of the three columns:
        - Iteration Number, negative logprobability of the data at the current iterations
        and convergence rate monitoring parameter (delta). Af first iteration, the 
        convergence rate is unknown and is thus denoted by NaN.
        ___
        Args:

        - logprob: (float) The logprob of the data as computed by the EM algorithm
        in the current iteration
        """

        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            if self.history:
                delta = torch.abs(delta)
                delta_rel = delta / torch.abs(self.history[-1])
            else:
                delta_rel = np.nan
            self.delta = delta
            self.delta_rel = delta_rel
            message = self._template.format(iter=self.iter+1,
                                            logprob=logprob,
                                            delta=self.delta_rel)
            print(message, file=sys.stdout)
            print("Convergence threshold:{}".format(self.tol))

        # History contains logprob of the data for the last 2 iterations
        self.history.append(logprob)
        self.iter += 1  # Increments the number of iterations

    @property
    def converged(self):
        if len(self.history) == 2:
            return self.delta_rel <= self.tol


class GenHMMclassifier(nn.Module):
    def __init__(self, mdlc_files=None, **options):
        """Initialize a model on CPU. Make sure to push to GPU at runtime."""
        super(GenHMMclassifier, self).__init__()

        if mdlc_files == None:
            self.nclasses = options["nclasses"]
            self.hmms = [GenHMM(**options) for _ in range(self.nclasses)]
            self.pclass = torch.ones(len(self.hmms))

        else:
            self.hmms = [load_model(fname) for fname in mdlc_files]
            self.pclass = torch.FloatTensor(
                [h.number_training_data for h in self.hmms])
            self.pclass = (self.pclass / self.pclass.sum())

    # consider do linear training based on GenHMMs

    def forward(self, x, weigthed=False):
        """compute likelihood of data under each GenHMM
        INPUT:
        x: The torch batch data
           or x should be tuple: (batch size * n_samples (padded length) * n_features, 
                                  tensor mask of the batch data)

        OUTPUT: tensor of likelihood, shape: data_size * ncl
        """

        if weigthed:
            batch_llh = [classHMM.pred_score(
                x) / classHMM.latestNLL for classHMM in self.hmms]
        else:
            batch_llh = [classHMM.pred_score(x) for classHMM in self.hmms]

        return torch.stack(batch_llh)

    def pushto(self, device):
        self.hmms = [h.pushto(device) for h in self.hmms]
        self.pclass = self.pclass.to(device)
        self.device = device
        return self

    def eval(self):
        for the_mdl in self.hmms:
            the_mdl.old_eval()
            the_mdl.eval()


class GenHMM(torch.nn.Module):
    def __init__(self, n_states=None, n_prob_components=None, device='cpu',
                 dtype=torch.FloatTensor,
                 EPS=1e-12, lr=None, em_skip=None,
                 net_H=28, net_D=14, net_nchain=10, mask_type="cross", p_drop=0.25,
                 startprob_type="first", transmat_type="random upper triangular",
                 tied_states=False):
        super(GenHMM, self).__init__()
        """ 
        Initialise parameters for NVP - HMM 
        ---
        Args:
        - n_states: Number of states in HMM
        - n_prob_components: No. of mixture components for mixture of normalizing flows
        - device: 'cpu'
        - dtype: torch.FloatTensor
        - EPS: epsilon (tolerance parameter)
        - lr: learning rate
        - em_skip: No. of iterations to skip before updating M-step 
        - net_H: no. of hidden channels
        - net_D: dimensionality of the input tensor
        - net_nchain: No. of coupling layers for RealNVP network
        - mask_type: Type of masking for RealNVP network
        - p_drop: Probability of dropout for enforcing regularization
        - startprob_type: "first" (first element non-zero, rest zero)
        - transmat_type: "random upper triangular" (Upper triangular matrix for transition probability)
        - tied_states: Flag to decide whether to use tied states for HMM

        Returns:
        - None ('self' object)
        """
        self.n_states = n_states
        self.dtype = dtype
        self.n_prob_components = n_prob_components

        self.device = device
        self.dtype = dtype
        self.EPS = EPS
        self.lr = lr
        self.em_skip = em_skip
        self.tied_states = tied_states

        # Initialize HMM parameters
        self.init_transmat(transmat_type)
        self.init_startprob(startprob_type)

        # Initialize generative model networks
        self.init_gen(H=net_H, D=net_D, nchain=net_nchain,
                      mask_type=mask_type, p_drop=p_drop)

        
        self._update_old_networks()
        
        self.old_eval()

        self.update_HMM = False

        # set the global_step
        self.global_step = 0

    def init_startprob(self, startprob_type="random"):
        """
        Initialize HMM initial coefficients.
        """
        if "random" in startprob_type:
            init = torch.abs(torch.randn(self.n_states))
            init /= init.sum()
            self.startprob_ = init

        elif "first" in startprob_type:
            init = torch.zeros(self.n_states)
            init[0] = 1
            self.startprob_ = init

        elif "uniform" in startprob_type:
            self.startprob_ = torch.ones(self.n_states)

        normalize(self.startprob_, axis=0)

        return self

    def init_transmat(self, transmat_type="random upper triangular"):
        """
        Initialize HMM transition matrix.
        """
        if "random" in transmat_type:
            self.transmat_ = (torch.randn(self.n_states, self.n_states)).abs()
        elif "uniform" in transmat_type:
            init = 1/self.n_states
            self.transmat_ = torch.ones(self.n_states, self.n_states) * init
        elif "triangular" in transmat_type:
            # use upper tra matrix
            self.transmat_ = (torch.randn(self.n_states, self.n_states)).abs()
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if i > j:
                        self.transmat_[i, j] = 0

        elif "ergodic" in transmat_type:
            self.transmat_ = torch.ones(self.n_states, self.n_states) \
                + torch.randn(self.n_states, self.n_states) * 0.01

        normalize(self.transmat_, axis=1)
        return self

    def init_gen(self, H, D, nchain, mask_type="cross", p_drop=0.25):
        """
        Initialize HMM probabilistic model as a mixture of normalizing flows (using RealNVP)
        ---
        Args:
        - H: number of hidden channels
        - D: number of input channels
        - nchain: number of coupling layers for RealNVP
        - mask_type: 
            - "chunk" : [[00000000, 11111111111],[1111111,  000000000000]]
            - "cross" : [[11111, 00000],[00000, 11111],[11111, 00000],[00000, 11111],...]
            - "conv" : NOTE:not implemented
        - p_drop: dropout probability

        Returns:
        - self (initialise the 'old_networks', 'networks')        
        """

        # Deciding to split the input channels into two parts for the RealNVP, 'd' denotes
        # the input dimensionality for each of the scaling (s) / translational (t) parameters
        d = D // 2

        # Defining the masks for the RealNVP flow
        # 'n_chain' defines the number of coupling layers for the RealNVP flow
        if mask_type == "chunk":
            masks = torch.from_numpy(
                np.array([[0]*d + [1]*(D-d), [1]*d + [0]*(D-d)] * nchain).astype(np.uint8))
        elif mask_type == "cross":
            masks = torch.from_numpy(
                np.array([[0, 1]*d, [1, 0]*d] * nchain).astype(np.uint8))
        elif mask_type == "conv":
            # To do
            pass

        # Check whether the mask is properly defined or not
        try:
            masks
        except NameError:
            print("masks are not defined")
            assert False

        # torch MultivariateNormal logprob gets error when input is cuda tensor
        # thus changing it to implementation
        prior = distributions.MultivariateNormal(torch.zeros(
            D).to(self.device), torch.eye(D).to(self.device))
        # prior = lambda x: GaussianDiag.logp(torch.zeros(D), torch.zeros(D), x)
        # self.flow = RealNVP(nets, nett, masks, prior)

        # A single feed-forward net that can be used for computing both s() and t().
        # The last layer therefore has dimension d + d = D (as d = D // 2)
        st_net = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(),
                                            nn.Linear(H, H), nn.LeakyReLU(),
                                            nn.Dropout(p_drop), nn.Linear(H, D))

        s_net = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(),
                                            nn.Linear(H, H), nn.LeakyReLU(),
                                            nn.Dropout(p_drop), nn.Linear(H, d))

        t_net = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(),
                                            nn.Linear(H, H), nn.LeakyReLU(),
                                            nn.Dropout(p_drop), nn.Linear(H, d))

        
        # Init mixture component weights - pi_{k,s}
        self.pi = self.dtype(np.random.rand(
            self.n_states, self.n_prob_components))
        
        # Should be that sum_{k=1}^{K} pi_{k,s} to 1
        normalize(self.pi, axis=1)

        # Compute the parameters logPIK_s = log(pi_{k,s})
        self.logPIk_s = self.pi.log()

        self.gens = RealNVP_FlowNets(self.n_states, self.n_prob_components, net_s=s_net, 
                                    net_t=t_net, net_st=st_net, masks=masks, prior=prior, 
                                    device=self.device, use_tied_states=self.tied_states)

        self.old_gens = RealNVP_FlowNets(self.n_states, self.n_prob_components, net_s=s_net, 
                                    net_t=t_net, net_st=st_net, masks=masks, prior=prior, 
                                    device=self.device, use_tied_states=self.tied_states)

        #TODO: Initialise the optimzier with the parameters
        
        if self.tied_states == True:
            learnable_params = sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in self.gens.networks.reshape(-1).tolist()], [])
            learnable_params_snet_shared = []
            
            #for s in range(self.n_states):
            #    s_net = self.gens.s_nets_per_state[s]
            #    learnable_params_snet_shared.append([p for p in s_net.parameters() if p.requires_grad==True])
            
            for k in range(self.n_prob_components):
                s_net = self.gens.s_nets_per_component[k]
                learnable_params_snet_shared.append([p for p in s_net.parameters() if p.requires_grad==True])

            learnable_params_snet_shared = sum(learnable_params_snet_shared, [])

            learnable_params_total = learnable_params + learnable_params_snet_shared
            self.optimizer = torch.optim.Adam(learnable_params_total, lr=self.lr)

        else:
            
            self.optimizer = torch.optim.Adam(
            sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in self.gens.networks.reshape(-1).tolist()], []), lr=self.lr)

        # Displaying the count of parameters in the model (gens.networks)
        total_num_trainable_params = self.count_params()
        print("The total number of trainable params:{}".format(total_num_trainable_params))

        return self

    def count_params(self):
        """
        Counts two types of parameters:
        ----
        Args:
        - networks(array of 'Flow' objects): A numpy array of flow networks which have to be used
        - shared_nets (list of nets): A python list of networks that are to be shared

        Returns:
        - total_num_trainable_params: Total no. of parameters in the model which are trainable 
        """
        if self.tied_states == False:
        
            learnable_params = sum([[p.numel() for p in flow.parameters() if p.requires_grad == True] for flow in self.gens.networks.reshape(-1).tolist()], [])
            total_num_trainable_params = sum(learnable_params)
            #total_num_params = sum(sum([[p.numel() for p in flow.parameters()] for flow in self.gens.networks.reshape(-1).tolist()], []))

        elif self.tied_states == True:
            
            learnable_params = sum([[p.numel() for p in flow.parameters() if p.requires_grad == True] for flow in self.gens.networks.reshape(-1).tolist()], [])
            learnable_params_snet_shared = []
            
            #for s in range(self.n_states):
            #    s_net = self.gens.s_nets_per_state[s]
            #    learnable_params_snet_shared.append([p.numel() for p in s_net.parameters() if p.requires_grad==True])
            
            for k in range(self.n_prob_components):
                s_net = self.gens.s_nets_per_component[k]
                learnable_params_snet_shared.append([p.numel() for p in s_net.parameters() if p.requires_grad==True])

            learnable_params_snet_shared = sum(learnable_params_snet_shared, [])

            learnable_params_total = learnable_params + learnable_params_snet_shared
            total_num_trainable_params = sum(learnable_params_total)

        return total_num_trainable_params

    def _update_old_networks(self):
        """ This function takes the 'self' object and copies the parameters and buffers from the 
        latest NVP-HMM output distributions into the old_version of the NVP-HMM output distributions
        ----
        Args:
        - self: object of GenHMM class

        Returns:
        - self: same object of GenHMM class, with attributes changed
        """
        for i in range(self.n_states):
            for j in range(self.n_prob_components):
                self.old_gens.networks[i, j].load_state_dict(
                    self.gens.networks[i,j].state_dict())
        return self

    def _affirm_networks_update(self):
        """ This function checks whether the parameters in 'old' output distributions for NVP-HMM
        and those in the current output distributions are same or not
        ----
        Args:
        - self: object of GenHMM class

        Returns:
        - self: same object of GenHMM class 
        """
        for i in range(self.n_states):
            for j in range(self.n_prob_components):
                state_dict = self.gens.networks[i, j].state_dict()
                for key, value in state_dict.items():
                    assert self.old_gens.networks[i, j].state_dict()[key] == value
        return self

    def pushto(self, device):
        """ Push the networks 
        ----
        Args:
            device ([str]): device to push the networks to 'cpu' or 'cuda:0'

        Returns:
            self : same object of the class
        """

        if self.tied_states == False:
            
            for s in range(self.n_states):
                for k in range(self.n_prob_components):
                    
                    # push new networks to device
                    self.gens.networks[s, k].to(device)
                    p = self.gens.networks[s, k].prior
                    self.gens.networks[s, k].prior = type(p)(p.loc.to(device),
                                                    p.covariance_matrix.to(device))

                    # push the old networks to device
                    self.old_gens.networks[s, k].to(device)
                    p = self.old_gens.networks[s, k].prior
                    self.old_gens.networks[s, k].prior = type(p)(p.loc.to(device),
                                                        p.covariance_matrix.to(device))
        elif self.tied_states == True:
            
            for s in range(self.n_states):
                
                # Push the shared s_nets_per_state of the old and new network to the device
                #self.gens.s_nets_per_state[s].to(device)
                #self.old_gens.s_nets_per_state[s].to(device)

                # Then push the networks as it is to the device (here the 'networks' contain only the t() network)
                # as the s() network should be shared and only passed separately during the llh computation
                for k in range(self.n_prob_components):
                    
                    # push new networks to device
                    self.gens.networks[s, k].to(device)
                    p = self.gens.networks[s, k].prior
                    self.gens.networks[s, k].prior = type(p)(p.loc.to(device),
                                                    p.covariance_matrix.to(device))

                    # push the old networks to device
                    self.old_gens.networks[s, k].to(device)
                    p = self.old_gens.networks[s, k].prior
                    self.old_gens.networks[s, k].prior = type(p)(p.loc.to(device),
                                                        p.covariance_matrix.to(device))
        
            for k in range(self.n_prob_components):
                
                # Push the shared s_nets_per_state of the old and new network to the device
                self.gens.s_nets_per_component[k].to(device)
                self.old_gens.s_nets_per_component[k].to(device)

        # Assign HMM variables to the device for updation
        self.startprob_ = self.startprob_.to(device) # Assign startprob to the device
        self.transmat_ = self.transmat_.to(device) # Assign transmat_ to the device
        self.pi = self.pi.to(device) # Assign pi to the device
        self.logPIk_s = self.logPIk_s.to(device) # Assign logPIk_s (\pi_{k,s}) to the device
        self.device = device # Assign device
        self.gens.device = device
        self.old_gens.device = device
        return self

    def old_eval(self):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # set old network mode as eval model
                self.old_gens.networks[s, k].eval()
        return self

    def eval(self):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # set mode as eval model
                self.gens.networks[s, k].eval()
        return self

    def train(self):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # set model as train mode
                self.gens.networks[s, k].train()
        return self

    def _initialize_sufficient_statistics(self):
        """Initializes sufficient statistics required for M-step.
        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.
        Returns
        -------
        nobs : int
            Number of samples in the data.
        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th
            state.
        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
        """

        stats = {'nobs': 0,
                 'nframes': 0,
                 'start': torch.zeros(self.n_states).to(self.device),
                 'trans': torch.zeros(self.n_states, self.n_states).to(self.device),
                 'mixture': torch.zeros(self.n_states, self.n_prob_components).to(self.device),
                 'loss': torch.FloatTensor([0]).to(self.device)
                 }

        self.stats = stats

    def _accumulate_sufficient_statistics(self, framelogprob, mask,
                                          posteriors, logprob, fwdlattice,
                                          bwdlattice, loglh_sk):
        """Updates sufficient statistics from a given sample.
        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.
        framelogprob : array, shape (batch_size, n_samples, n_components)
            Log-probabilities of each sample under each of the model states.
        posteriors : array, shape (batch_size, n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.
        fwdlattice, bwdlattice : array, shape (batch_size, n_samples, n_components)
            Log-forward and log-backward probabilities.
        loglh_sk : array, shape (batch_size, n_samples, n_components, n_prob_components)
            Log-probabilities of each batch sample under each components of each states.
        """

        batch_size, n_samples, n_components = framelogprob.shape

        self.stats['nframes'] += mask.sum()
        self.stats['nobs'] += batch_size
        self.stats['start'] += posteriors[:, 0].sum(dim=0)

        log_xi_sum = _compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                         torch.log(self.transmat_ + self.EPS),
                                         bwdlattice, framelogprob,
                                         torch.ones(batch_size,
                                                    n_components,
                                                    n_components,
                                                    device=self.device) * float('-inf'),
                                         logprob, mask)
        # _log_xi_sum = _compute_log_xi_sum(n_samples, n_components,\
        #                                    self.dtype(fwdlattice).to(self.device),\
        #                                    self.dtype(log_mask_zero(self.transmat_)).to(self.device),\
        #                                    self.dtype(bwdlattice).to(self.device), \
        #                                    self.dtype(framelogprob).to(self.device),\
        #                                    self.dtype(np.full((n_components, n_components), -np.inf)).to(self.device))

        self.stats['trans'] += torch.exp(log_xi_sum).sum(0)

 #       print(loglh_sk.shape, self.n_states, self.n_prob_components)
        # max_loglh = torch.max(torch.max(loglh_sk, dim=1)[0],dim=1)[0]

        local_loglh_sk = loglh_sk
        max_loglh = torch.max(local_loglh_sk, dim=3, keepdim=True)[0]
        # should be careful to use the minus max trick here
        gamma_ = self.pi.reshape(1, 1, self.n_states, self.n_prob_components) * \
            (local_loglh_sk - max_loglh).exp()
        gamma_ = gamma_ / (gamma_.sum(3, keepdim=True) + self.EPS)
        # # set the elements corresponding to padded values to be zeros, this is done by zeroes in posteriors
        gamma = posteriors.unsqueeze(dim=3) * gamma_

        # In- line test for gamma computation, set the if condition to be true to compare gamm and statcs_prob_components
        if False:
            statcs_prob_components = torch.zeros(
                batch_size, n_samples, self.n_states, self.n_prob_components, device=self.device)
    #        print(max_loglh.shape)

            gamma_ = torch.zeros(
                batch_size, n_samples, self.n_states, self.n_prob_components, device=self.device)
            for b in range(batch_size):
                for n in range(n_samples):
                    for i in range(self.n_states):
                        for k in range(self.n_prob_components):
                            gamma_[b, n, i, k] = self.pi[i, k] * \
                                (local_loglh_sk[b, n, i, k] -
                                 local_loglh_sk[b, n, i, :].max()).exp()

                        gamma_[b, n, i, :] = gamma_[b, n, i, :] / \
                            (gamma_[b, n, i, :].sum() + self.EPS)

                        statcs_prob_components[b, n, i,
                                               :] = posteriors[b, n, i] * gamma_[b, n, i, :]

        self.stats["mixture"] += gamma.sum(1).sum(0)

        return

    def _do_forward_pass(self, framelogprob, mask):
        batch_size, n_samples, n_components = framelogprob.shape
        # in case log computation encounter log(0), do log(x + self.EPS)

        # To Do: matain hmm parameters as torch tensors
        log_startprob = torch.log(self.startprob_ + self.EPS)
        log_transmat = torch.log(self.transmat_ + self.EPS)

        return _forward(n_samples, n_components, log_startprob,
                        log_transmat, framelogprob, mask)

    def _do_backward_pass(self, framelogprob, mask):
        batch_size, n_samples, n_components = framelogprob.shape

        # To Do: matain hmm parameters as torch tensors
        log_startprob = torch.log(self.startprob_ + self.EPS)
        log_transmat = torch.log(self.transmat_ + self.EPS)
        return _backward(n_samples, n_components, log_startprob,
                         log_transmat, framelogprob, mask)

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        # Normalizes the input array so that the exponent of the sum is 1
        lse_gamma = torch.logsumexp(log_gamma, dim=2)

        log_gamma -= lse_gamma[:, :, None]

        return torch.exp(log_gamma)

    def pred_score(self, X):
        """ Update the base score method, such that the scores of sequences are returned
        score: the log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        Returns
        -------
        logprob : list of floats, [logprob1, logprob2, ... ]
            Log likelihood of ``X``.
        """
        # now mask is used, need to pass mask as well
        # will consider to do batch as well in testig
        # mask = torch.ones(1, lengths[0], dtype=torch.uint8)
        # X = self.dtype(X[None,:]).to(self.device)
        logprob = self.forward(X, testing=True)
        return logprob
    
    def forward(self, batch, testing=False):
        """PYTORCH FORWARD, NOT HMM forward algorithm. This function is called for each batch.
        Input: batch of sequences, array size, (batch_size, n_samples, n_dimensions)
        Output: Loss, scaler
        """

        if self.update_HMM and not testing:
            self._initialize_sufficient_statistics()

        x, x_mask = batch
        batch_size = x.shape[0]
        n_samples = x.shape[1]

        # get the log-likelihood for posterior computation
        with torch.no_grad():
            # Two posteriors to be computed here:
            # 1. the hidden state posterior, post
            #NOTE: Old version - old_llh, old_loglh_sk = self._getllh(self.old_gens.networks, batch)
            old_llh, old_loglh_sk = self.old_gens._getllh(self.logPIk_s, batch)
            old_llh[~x_mask] = 0
            old_logprob, old_fwdlattice = self._do_forward_pass(
                old_llh, x_mask)
            # assert ((old_logprob <= 0).all())

            if testing:
                # each EM step sync old_networks and networks, so it is ok to test on old_networks
                return old_logprob

            old_bwdlattice = self._do_backward_pass(old_llh, x_mask)
            posteriors = self._compute_posteriors(
                old_fwdlattice, old_bwdlattice)

            posteriors[~x_mask] = 0
            post = posteriors

            # 2. the probability model components posterior, k condition on hidden state, observation and hmm model
            # Compute log-p(chi | s, X) = log-P(X|s,chi) + log-P(chi|s) - log\sum_{chi} exp ( log-P(X|s,chi) + log-P(chi|s) )

            log_num = old_loglh_sk.detach() + self.logPIk_s.reshape(1,
                                                                    self.n_states, self.n_prob_components)
            #log_num = brackets.detach()
            log_denom = torch.logsumexp(log_num, dim=3)

            logpk_sX = log_num - \
                log_denom.reshape(batch_size, n_samples, self.n_states, 1)
            # To Do: normalize logpk_sX before set un-masked values
            logpk_sX[~x_mask] = 0

        # hmm parameters should be updated based on old model
        if self.update_HMM and not testing:
            self._accumulate_sufficient_statistics(old_llh, x_mask,
                                                   posteriors, old_logprob,
                                                   old_fwdlattice, old_bwdlattice, old_loglh_sk)

        # Get the log-likelihood to format cost such self.gens.networks such it can be optimized
        #NOTE: Old version - llh, self.loglh_sk = self._getllh(self.gens.networks, batch)
        llh, self.loglh_sk = self.gens._getllh(self.logPIk_s, batch)
        # compute sequence log-likelihood in self.gens.networks, just to monitor the self.gens.networks performance
        with torch.no_grad():
            llh[~x_mask] = 0
            logprob, _ = self._do_forward_pass(llh, x_mask)
        # assert((logprob <= 0).all())
        # Brackets = log-P(X | chi, S) + log-P(chi | s)
        brackets = torch.zeros_like(self.loglh_sk)
        # Todo: implement update pi_s_k ?
        brackets[x_mask] = self.loglh_sk[x_mask] + \
            self.logPIk_s.reshape(1, self.n_states, self.n_prob_components)

        #  The .sum(3) call sums on the components and .sum(2).sum(1) sums on all states and samples
        # loss = -(post * (torch.exp(logpk_sX) * brackets).sum(3)).sum(2).sum(1).sum()/float(x_mask.sum())
        loss = -(post[x_mask] * (torch.exp(logpk_sX) * brackets)
                 [x_mask].sum(2)).sum()/float(batch_size)
        return loss, logprob.sum()

    def fit(self, traindata):
        """Performs one EM step and `em_skip` backprops before returning. The optimizer is re-initialized after each EM step.
            Follow the loss in stderr
            Input : traindata : torch.data.DataLoader object wrapping the batches.
            Output : None
        """
        # get the adaptive learning rate
        ada_lr = step_learning_rate_decay(init_lr=self.lr,
                                          global_step=self.global_step,
                                          minimum=1e-4,
                                          anneal_rate=0.98)

        # Assigning the adaptive learning rate to the trainable optim parameters
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = ada_lr

        # Loads the optimizer state so that changes in lr are registered
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # total number of sequences
        n_sequences = len(traindata.dataset)

        # Measure epoch time
        starttime = timer()

        ##############################################################################
        # Training process begins here
        ##############################################################################

        for i in range(self.em_skip):
            # if i is the index of last loop, set update_HMM as true

            if i == self.em_skip-1:
                self.update_HMM = True
            else:
                self.update_HMM = False

            total_loss = 0
            total_logprob = 0
            for b, data in enumerate(traindata):
                # start = dt.now()
                self.optimizer.zero_grad()
                loss, logprob_ = self.forward(data, testing=False)
                loss.backward()

                self.optimizer.step()
                total_loss += loss.detach().data
                total_logprob += logprob_

            # consider put a stop criteria here to

            print("epoch:{}\tclass:{}\tStep:{}\tb:{}\tLoss:{}\tNLL:{}".format(self.iepoch,
                                                                              self.iclass, i, b,
                                                                              total_loss /
                                                                              (b+1),
                                                                              -total_logprob/n_sequences), file=sys.stdout)

        ################################################################################################
        # Updating the HMM parameters such as start prob vector, transmat and mixture of weights for the
        # generative model
        ################################################################################################

        # Perform EM step
        # Update initial probabs
        # startprob_ = self.startprob_prior - 1.0 + self.stats['start']
        startprob_ = self.stats['start']
        self.startprob_ = torch.where(self.startprob_ == 0.0,
                                      self.startprob_, startprob_)
        normalize(self.startprob_, axis=0)

        # Update transition
        # transmat_ = self.transmat_prior - 1.0 + self.stats['trans']
        transmat_ = self.stats['trans']
        self.transmat_ = torch.where(self.transmat_ == 0.0,
                                     self.transmat_, transmat_)
        normalize(self.transmat_, axis=1)

        # Update prior
        tmp_pi = self.pi.clone()
        self.pi = self.stats["mixture"]
        normalize(self.pi, axis=1)
        # In case we get a line of zeros in the stats, skip the update brought by self.stats["mixture"]
        correct_idx = self.pi.sum(1).isclose(
            torch.ones(1, device=self.pi.device))
        self.pi[~ correct_idx] = tmp_pi[~ correct_idx]

        # any zero element in self.pi would cause -inf in self.logPIk_s. Fix: replace with self.EPS
        zero_idx = self.pi.isclose(torch.zeros(1, device=self.pi.device))
        self.pi[zero_idx] = self.EPS
        # normalize again
        normalize(self.pi, axis=1)
        # get log of pi
        self.logPIk_s = self.pi.log()

        # update output probabilistic model, networks here
        self._update_old_networks()
        self.old_eval()

        # store the latest NLL of the updated GenHMM model
        log_p_all = torch.cat(list(map(self.pred_score, traindata)))
        self.latestNLL = -log_p_all.sum()/n_sequences
        # store the average lop_p
        self.avrg_log_p = log_p_all.reshape(1, -1).squeeze().logsumexp(0)
        # stoe the max log_p
        self.max_log_p = log_p_all.max()

        # This is the training NLL that is going to be checked for convergence at the end of these classes
        print("epoch:{}\tclass:{}\tLatest NLL:\t{}".format(
            self.iepoch, self.iclass, self.latestNLL), file=sys.stdout)

        # Epoch time measurement ends here
        endtime = timer()

        # Measure wallclock time
        print("Time elapsed measured in seconds:{}".format(endtime - starttime))

        # Inserting convergence check here
        self.monitor_.report(self.latestNLL)

        # Flag back to False
        self.update_HMM = False
        # set global_step
        self.global_step += 1

        # Break off if model has converged which is set by the convergence flag
        if self.monitor_.converged == True:
            print("Convergence attained!!")
            return True
        else:
            return False


class wrapper(torch.nn.Module):
    def __init__(self, mdl):
        super(wrapper, self).__init__()
        self.userdata = mdl


def save_model(mdl, fname=None):
    torch.save(wrapper(mdl), fname)
    return 0


def load_model(fname):
    """Loads a model on CPU by default."""
    savable = torch.load(fname, map_location='cpu')
    return savable.userdata


def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis, keepdim=True)
    a_sum[a_sum == 0] = 1
    a /= a_sum
