import os
os.add_dll_directory("C:/Users/narde/.mujoco/mujoco200/bin")
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
import math
from torch.utils.tensorboard import SummaryWriter
import os
import pickle


# DO NOT TOUCH
# CHANGING MUJOCO TO RUN CONTINUAL ENV
import continualworld.envs as cw_envs
import metaworld.envs.mujoco.env_dict as env_dict
from continualworld.tasks import TASK_SEQS
import metaworld
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

original_mujoco_init = MujocoEnv.__init__
def patched_mujoco_init(self, *args, **kwargs):
    original_mujoco_init(self, *args, **kwargs)
    self.max_path_length = 9999999
    class MujocoDataProxy:
        def __init__(self, data, model):
            self._data = data
            self._model = model
        def __getattr__(self, name):
            return getattr(self._data, name)
        def __setattr__(self, name, value):
            if name in ['_data', '_model']:
                super().__setattr__(name, value)
            else:
                setattr(self._data, name, value)
        def get_joint_qpos(self, name):
            addr = self._model.get_joint_qpos_addr(name)
            if hasattr(addr, '__iter__'): return self._data.qpos[addr[0]:addr[1]]
            return self._data.qpos[int(addr)]
        def get_joint_qvel(self, name):
            addr = self._model.get_joint_qvel_addr(name)
            if hasattr(addr, '__iter__'): return self._data.qvel[addr[0]:addr[1]]
            return self._data.qvel[int(addr)]
        def set_joint_qpos(self, name, value):
            addr = self._model.get_joint_qpos_addr(name)
            if hasattr(addr, '__iter__'): self._data.qpos[addr[0]:addr[1]] = value
            else: self._data.qpos[int(addr)] = value
        def set_joint_qvel(self, name, value):
            addr = self._model.get_joint_qvel_addr(name)
            if hasattr(addr, '__iter__'): self._data.qvel[addr[0]:addr[1]] = value
            else: self._data.qvel[int(addr)] = value
            
    self.data = MujocoDataProxy(self.sim.data, self.model)

original_do_simulation = MujocoEnv.do_simulation
def patched_do_simulation(self, *args, **kwargs):
    self.max_path_length = 9999999
    return original_do_simulation(self, *args, **kwargs)
MujocoEnv.do_simulation = patched_do_simulation

MujocoEnv.__init__ = patched_mujoco_init

if hasattr(cw_envs.MT50, '_train_classes'):
    cw_envs.MT50._train_classes.update(env_dict.ALL_V2_ENVIRONMENTS)
cw_envs.MT50.train_classes.update(env_dict.ALL_V2_ENVIRONMENTS)

original_get_subtasks = cw_envs.get_subtasks
def patched_get_subtasks(task_name):
    if task_name.endswith('-v2'):
        v1_name = task_name.replace('-v2', '-v1')
        v1_tasks = original_get_subtasks(v1_name)
        v2_env_cls = env_dict.ALL_V2_ENVIRONMENTS[task_name]
        
        v2_tasks = []
        for t in v1_tasks:
            data_dict = pickle.loads(t.data)
            exact_v2_payload = {
                'env_cls': v2_env_cls,
                'rand_vec': data_dict['rand_vec'],
                'partially_observable': data_dict['partially_observable']
            }
            new_data_binary = pickle.dumps(exact_v2_payload)
            new_t = metaworld.Task(env_name=task_name, data=new_data_binary)
            v2_tasks.append(new_t)
        return v2_tasks
    return original_get_subtasks(task_name)

cw_envs.get_subtasks = patched_get_subtasks


# creating the env with v2 versions (more stable rewards)
CW10_V2 = [task.replace("-v1", "-v2") for task in TASK_SEQS["CW10"]]

env = cw_envs.get_cl_env(
    tasks=CW10_V2[:4],
    steps_per_task=400_000
)


writer = SummaryWriter(log_dir="runs/SAC_ProtoCRL")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# CONSTANTS
N_ACTIONS = env.action_space.shape[0]
N_OBS = env.observation_space.shape[0]
Q_lr = 3e-4
ACT_lr = 3e-4
GMM_lr = 3e-4
BUFFER_SIZE = 50000
SAMPLE_LENGTH = 256
GAMMA = 0.99
TAU = 0.005
target_network_frequency = 2
protocrl_frequency = 2
WARM_GET_TRANSITIONS = 10000
WARM_TRAIN_EPOCHS = 200
K = 7                       # number of means
ALPHA_LOGP = 0.1
ALPHA_KL = 0.001
LAMBDA = 1  
B_E = 2.35
B_H = 5
POST_TEMP = 2.0             # used to soften the loglikelihood
alpha = 0.2                 # SAC's fixed entropy coefficient
NUM_TASKS = 4               # number of tasks to run
performance_matrix = np.zeros((NUM_TASKS, NUM_TASKS))           # performance_matrix[i, j] = success rate on task j after training on task i


# CREATE NETWORKS
class SoftQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS + N_ACTIONS, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5
LOG_STD_MIN_GMM = 0



# the actor is composed of an encoder part which is only updated from the actor, then GMM head and the rest of the actor network.
class ProtoCRL_Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.encoder = nn.Sequential(
                    nn.Linear(N_OBS, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
        )

        # actor parameters
        self.fc_mean = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS)
        )
        self.fc_logstd = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS)
        )
        
        # GMM parameters
        self.mu = nn.Parameter(torch.empty(K, 256).uniform_(-0.5, 0.5))
        self.log_sigma = nn.Parameter(torch.full((K, 256), 0.0))
        self.pi = nn.Parameter(torch.zeros(K))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, step):
        h_raw = self.encoder(x)
        # forcing the encoder's output onto a massive sphere with a radius of 20.0: it's big enough to not make clusters overlap and small enough to not make them drift far away bringing collapse
        h = F.normalize(h_raw, p=2, dim=-1) * 20.0          
        
        # actor forward
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  

        # GMM forward
        clamped_log_sigma = torch.clamp(self.log_sigma, min=LOG_STD_MIN_GMM, max=LOG_STD_MAX)              # clamping sigma to not exploit elbo
        bounded_mu = self.mu   
        h_gmm = h.detach()                                          # do not pass GMM's grad in the encoder

        # OBJECTIVE: GET COMPONENTS FOR THE ELBO LOSS
        # computing the likelihood p(X | teta) = N(X, teta) since we consider teta the "right" parameters. In particular we need log(likelihood) = log(N(X, teta))
        diff = h_gmm.unsqueeze(1) - bounded_mu.unsqueeze(0)         
        mahalanobis = (diff ** 2) / torch.exp(clamped_log_sigma.unsqueeze(0))              # the covariance matrix is (by hyphotesis) diagonal making the inverse into 1/sigma
        log_det = clamped_log_sigma.sum(dim=-1).unsqueeze(0)   
        log_likelihood = -0.5 * (log_det + mahalanobis.sum(dim=-1))
        
        # the prior is p(teta) in the case of a GMM the prob of having that parameters is given by the distribution pi
        log_prior = F.log_softmax(self.pi, dim=0).unsqueeze(0)                                     # taking the softmax ensures that the sum over pi = 1, we take the log for math reason in the following formula
        
        # the posterior is q(teta | X) which is proportional to prior * likelihood in log terms it becomes a sum                  
        log_posterior = F.log_softmax(log_prior + log_likelihood / POST_TEMP, dim=-1)             # taking softmax since the posterior must be a distribution      
        posterior = torch.exp(log_posterior)            

        if step is not None and step % 49 == 0:
            writer.add_scalar("GMM/log_sigma_min", clamped_log_sigma.min().item(), step)
            writer.add_scalar("GMM/log_sigma_mean", clamped_log_sigma.mean().item(), step)    
            writer.add_scalar("GMM/mu_norm_mean", self.mu.norm(dim=-1).mean().item(), step)
            writer.add_scalar("GMM/bounded_mu_norm_mean", bounded_mu.norm(dim=-1).mean().item(), step)
            writer.add_scalar("GMM/h_norm_mean", h.norm(dim=-1).mean().item(), step)         
            writer.add_scalar("Encoder/h_batch_std", h.std(dim=0).mean().item(), step)  # average per-dim std across the batch         

            with torch.no_grad():
                # tracking raw distances by synnubg across the 256 dimensions first to get total distance per cluster (Shape: batch, 7)
                cluster_dists = mahalanobis.sum(dim=-1) 
                min_dist, closest_ids = cluster_dists.min(dim=-1)
                max_dist, furthest_ids = cluster_dists.max(dim=-1)
                writer.add_scalar("Distances/Mahalanobis_Closest_Cluster", min_dist.mean(), step)
                writer.add_scalar("Distances/Mahalanobis_Furthest_Cluster", max_dist.mean(), step)
                writer.add_scalar("Distances/Mahalanobis_Closest_Cluster_ID", closest_ids[0].item(), step)
                writer.add_scalar("Distances/Mahalanobis_Furthest_Cluster_ID", furthest_ids[0].item(), step)
                
                # tracking likelihood distance (raw distance + variance)
                max_ll, log_winner_ids = log_likelihood.max(dim=-1)
                min_ll, log_loser_ids = log_likelihood.min(dim=-1)
                writer.add_scalar("Distances/LogLikelihood_Winner", max_ll.mean(), step)
                writer.add_scalar("Distances/LogLikelihood_Loser", min_ll.mean(), step)       
                writer.add_scalar("Distances/LogLikelihood_Winner_ID", log_winner_ids[0].item(), step)
                writer.add_scalar("Distances/LogLikelihood_Loser_ID", log_loser_ids[0].item(), step)                   
        
        return mean, log_std, posterior, log_posterior, log_likelihood, log_prior

    def get_action(self, x, step=None):
        mean, log_std, posterior, log_posterior, log_likelihood, log_prior = self(x, step)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(self.action_scale) + 2 * (math.log(2) - x_t - F.softplus(-2 * x_t))
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, posterior, log_posterior, log_likelihood, log_prior


# INSTANTIATE CLASSES
actor = ProtoCRL_Actor(env).to(device)
actor_params = list(actor.encoder.parameters()) + list(actor.fc_mean.parameters()) + list(actor.fc_logstd.parameters())
gmm_params = [actor.mu, actor.log_sigma, actor.pi] 
actor_optimizer = torch.optim.Adam(actor_params, lr=ACT_lr)
gmm_optimizer = torch.optim.Adam(gmm_params, lr=GMM_lr)

qf1 = SoftQNetwork().to(device)
qf2 = SoftQNetwork().to(device)
qf1_target = SoftQNetwork().to(device)
qf2_target = SoftQNetwork().to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), Q_lr)


# CREATE PARTIONED REPLAY BUFFER
@dataclass
class ReplayBufferSamples:
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer:
    def __init__(self):
        self.observations = np.zeros((K+1, BUFFER_SIZE, N_OBS), dtype=np.float32)
        self.next_observations = np.zeros((K+1, BUFFER_SIZE, N_OBS), dtype=np.float32)
        self.actions = np.zeros((K+1, BUFFER_SIZE, N_ACTIONS), dtype=np.float32)
        self.rewards = np.zeros((K+1, BUFFER_SIZE), dtype=np.float32)
        self.dones = np.zeros((K+1, BUFFER_SIZE), dtype=np.float32)
        self.pos = np.zeros(K+1, dtype=int)
        self.full = np.zeros(K+1, dtype=bool)

    def add(self, obs, next_obs, action, reward, done, k):
        self.observations[k, self.pos[k]] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.next_observations[k, self.pos[k]] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.actions[k, self.pos[k]] = action
        self.rewards[k, self.pos[k]] = float(reward)
        self.dones[k, self.pos[k]] = bool(done)

        self.pos[k] += 1
        if self.pos[k] == BUFFER_SIZE:
            self.full[k] = True
            self.pos[k] = 0

    def sample(self, batch_size: int, table0_fraction: float = 0.5, step = None) -> ReplayBufferSamples:
        old_tables = np.where(self.full | (self.pos > 5000))[0]             # sampling only from tables with at least 5k elements
        old_tables = old_tables[old_tables != 0]                            # exlude default table for now

        if step is not None and step % 2000 == 0:
            writer.add_scalar("ReplayBuffer/num_old_tables", len(old_tables), step)

        n_table0 = int(batch_size * table0_fraction)                        # sampling a different portion of default table compared to other tables
        n_old = batch_size - n_table0

        upper0 = BUFFER_SIZE if self.full[0] else self.pos[0]
        k_inds = [0] * n_table0
        batch_inds = list(np.random.randint(0, upper0, size=n_table0))

        if len(old_tables) > 0:
            samples_per_table = n_old // len(old_tables)
            remainder = n_old % len(old_tables)
            for i, k in enumerate(old_tables):
                current_samples = samples_per_table + (1 if i < remainder else 0)
                if current_samples == 0:
                    continue
                upper_bound = BUFFER_SIZE if self.full[k] else self.pos[k]
                inds = np.random.randint(0, upper_bound, size=current_samples)
                k_inds.extend([k] * current_samples)
                batch_inds.extend(inds)
        else:
            extra = np.random.randint(0, upper0, size=n_old)   # no old clusters yet (early task 0) — fill from table 0
            k_inds.extend([0] * len(extra))
            batch_inds.extend(extra)

        return ReplayBufferSamples(
            observations=torch.as_tensor(self.observations[k_inds, batch_inds]).to(device),
            actions=torch.as_tensor(self.actions[k_inds, batch_inds]).to(device),
            next_observations=torch.as_tensor(self.next_observations[k_inds, batch_inds]).to(device),
            dones=torch.as_tensor(self.dones[k_inds, batch_inds]).to(device),
            rewards=torch.as_tensor(self.rewards[k_inds, batch_inds]).to(device),
        )
    

# each tasks has different reward functions (with different magnitudes of rewards) making SAC crash
class RunningRewardScaler:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.S = 0.0

    def normalize(self, reward):
        self.n += 1
        old_mean = self.mean
        self.mean += (reward - self.mean) / self.n
        self.S += (reward - old_mean) * (reward - self.mean)
        
        if self.n > 1:
            variance = self.S / (self.n - 1)
            std = math.sqrt(variance)
        else:
            std = 1.0
            
        return (reward - self.mean) / (std + self.epsilon)


def compute_TDLoss(data, step=None):
    with torch.no_grad():
        next_state_actions, next_state_log_pi, *_ = actor.get_action(data.next_observations, step)
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * GAMMA * (min_qf_next_target).view(-1)
    
    qf1_a_values = qf1(data.observations, data.actions).view(-1)
    qf2_a_values = qf2(data.observations, data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    if step is not None and step % 5 == 0:
        writer.add_scalar("Loss/TDLoss", qf_loss.item(), step)
        writer.add_scalar("Metrics/Qmaxvalue", max(qf1_a_values.abs().max().item(), qf2_a_values.abs().max().item()), step)

    return qf_loss        


def compute_ActorLoss(data, step=None):
    pi, log_pi, _, posterior, log_posterior, log_likelihood, log_prior = actor.get_action(data.observations, step)
    qf1_pi = qf1(data.observations, pi)
    qf2_pi = qf2(data.observations, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

    return actor_loss, posterior, log_posterior, log_likelihood, log_prior


def compute_ProtoCRL_Loss(data, step = None, actorLoss = None, posterior = None, log_posterior = None, log_likelihood = None, log_prior = None):
    if step is None:
        actorLoss, posterior, log_posterior, log_likelihood, log_prior = compute_ActorLoss(data, step)

    # posterior is of shape [batch_size, K]
    # log(p(x)) = sum (probability of being in that gaussian (posterior) * value we have (likelihood))  following the definition of expected value
    log_p = (posterior * log_likelihood).sum(dim=-1).mean()
    # KL divergence sum(q(teta | X) * (log(q(teta | X) - log(p(teta))))
    kl = (posterior * (log_posterior - log_prior)).sum(dim=-1).mean()
    L_elbo = - ALPHA_LOGP * log_p + ALPHA_KL * kl 

    m = posterior.mean(dim=0)
    L_entropy = (m * torch.log(m + 1e-6)).sum()

    hoyer = ((math.sqrt(K) * torch.norm(posterior, 1, dim=-1)) / (torch.norm(posterior, 2, dim=-1) + 1e-6)) - 1
    L_hoyer = hoyer.mean()

    L_ProtoCRL = actorLoss.detach() + LAMBDA * (L_elbo + B_E * L_entropy + B_H * L_hoyer)

    if step is not None and step % 5 == 0:
        writer.add_scalar("Loss/ActorLoss", actorLoss.item(), step)
        writer.add_scalar("Loss/ELBO", L_elbo.item(), step)
        writer.add_scalar("Loss/Entropy", L_entropy.item(), step)
        writer.add_scalar("Loss/Hoyer", L_hoyer.item(), step)
        writer.add_scalar("Loss/ProtoCRL", L_ProtoCRL.item(), step)

    if step is not None and step % 50 == 0:
        writer.add_histogram("Distributions/Posterior", posterior, step)
        writer.add_scalar("Distributions/Max_Probability", posterior.max(dim=-1)[0].mean(), step)
        with torch.no_grad():
            effective_k = torch.exp(-L_entropy)                     
        writer.add_scalar("Clusters/Effective_Num_Used", effective_k.item(), step)
        with torch.no_grad():
            pi_probs = F.softmax(actor.pi, dim=0)
            for i in range(K):
                writer.add_scalar(f"Prior_Prob/Cluster_{i+1}", pi_probs[i].item(), step)

    return L_ProtoCRL


# initializes means to be as sparse as possible
@torch.no_grad()
def init_gmm_from_latents(obs_np, n=10):
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    h = F.normalize(actor.encoder(x), dim=-1) * 20.0
    centers = h[torch.randint(len(h), (1,), device=device)]             # choose the first center at random
    for _ in range(K - 1):
        # for each cluster we search its center based on the distance from the others. We choose the center based on a multinomial so that we have more prob to have it farther than others
        d2 = torch.cdist(h, centers).pow(2).min(dim=1).values
        centers = torch.cat([centers, h[torch.multinomial(d2 + 1e-8, 1)]], dim=0)
    # using k-means to move the centers in the mean of its h's
    for _ in range(n):
        assign = torch.cdist(h, centers).argmin(dim=1)
        for k in range(K):
            m = assign == k
            if m.any():
                centers[k] = h[m].mean(dim=0)
    actor.mu.data.copy_(centers)


# TRAINING LOOP
def ProtoCRL_SAC():
    global alpha
    rb = ReplayBuffer()
    global_step = 0
    reward_scaler = RunningRewardScaler()
    task_cluster_counts = np.zeros((NUM_TASKS, K+1), dtype=np.int64)
    print("Warming up...")

    warmup_env = cw_envs.get_single_env(task=CW10_V2[0], one_hot_idx=0, one_hot_len=NUM_TASKS)
    state = warmup_env.reset()
    for _ in range(WARM_GET_TRANSITIONS): 
        action, _, _, posterior, *_ = actor.get_action(torch.FloatTensor(state).unsqueeze(0).to(device), global_step)
        action = action[0].detach().cpu().numpy()
        next_state, raw_reward, done, info = warmup_env.step(action)
        reward = reward_scaler.normalize(raw_reward)
        is_true_terminal = False

        if info.get("success", False):
            success = True

        max_prob, k_idx = posterior.max(dim=-1)
        k = k_idx.item() + 1
        rb.add(state, next_state, action, reward, is_true_terminal, 0)          
        
        if done:
            state = warmup_env.reset()
        else:
            state = next_state

    init_gmm_from_latents(rb.observations[0, :WARM_GET_TRANSITIONS])
    # send data into the buffers after we initialized the means
    with torch.no_grad():
        obs_all = torch.as_tensor(rb.observations[0, :WARM_GET_TRANSITIONS], device=device)
        _, _, _, post, *_ = actor.get_action(obs_all, None)
    ks = post.argmax(dim=-1).cpu().numpy() + 1
    for i, k in enumerate(ks):
        rb.add(rb.observations[0, i], rb.next_observations[0, i], rb.actions[0, i],
            rb.rewards[0, i], rb.dones[0, i], int(k))

    for _ in range(WARM_TRAIN_EPOCHS):
        data = rb.sample(SAMPLE_LENGTH)                   # let's warm up on a smaller number
        L_ProtoCRL = compute_ProtoCRL_Loss(data)

        gmm_optimizer.zero_grad()
        L_ProtoCRL.backward()
        torch.nn.utils.clip_grad_norm_(gmm_params, max_norm=1.0)
        gmm_optimizer.step()

    print("Starting training...")

    pbar = tqdm(total=env.steps_limit, desc="Solving tasks")
    episode_counter = 0
    while global_step < env.steps_limit:
        state = env.reset()
        action, _, _, posterior, *_ = actor.get_action(torch.FloatTensor(state).unsqueeze(0).to(device), global_step)
        action = action[0].detach().cpu().numpy()
        k = posterior.argmax().item() + 1               # choose best cluster and add 1 since 0 is the default table       
        ep_reward = 0
        done = False
        success = False

        while not done:
            prev_seq_idx = env.cur_seq_idx
            next_state, raw_reward, done, info = env.step(action)
            reward = reward_scaler.normalize(raw_reward)
            is_true_terminal = False 
            
            if info.get("success", False):
                success = True

            if global_step % 50 == 0:
                writer.add_scalar("Raw/raw_reward_step", raw_reward, global_step)
                writer.add_scalar("Raw/reward_step", reward, global_step)
            ep_reward += reward

            if env.cur_seq_idx != prev_seq_idx:           
                completed_task_id = prev_seq_idx
                tqdm.write(f"Task {completed_task_id} completed, evaluating on past tasks...")
                reward_scaler.reset()
                evaluate_past_tasks(completed_task_id, global_step)

            max_prob, k_idx = posterior.max(dim=-1)
            k = k_idx.item() + 1

            task_cluster_counts[min(env.cur_seq_idx, NUM_TASKS - 1), k] += 1
            if global_step % 5000 == 0:
                row_sums = task_cluster_counts.sum(axis=1, keepdims=True).clip(min=1)
                normalized = task_cluster_counts / row_sums
                for t in range(NUM_TASKS):
                    writer.add_scalars(
                        f"TaskClusterMap/task_{t}",
                        {f"cluster_{k}": normalized[t, k] for k in range(K + 1)},
                        global_step,
                    )
                task_cluster_counts[:] = 0  

            rb.add(state, next_state, action, reward, is_true_terminal, 0)             
            rb.add(state, next_state, action, reward, is_true_terminal, k)

            if rb.pos[0] > SAMPLE_LENGTH:                                
                data = rb.sample(SAMPLE_LENGTH, step=global_step)

                # update Q
                TDloss = compute_TDLoss(data, global_step)
                q_optimizer.zero_grad()
                TDloss.backward()
                torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=1.0)
                q_optimizer.step()

                # update actor only on actorloss
                actor_loss, posterior, log_posterior, log_likelihood, log_prior = compute_ActorLoss(data, global_step)
                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph = global_step % protocrl_frequency == 0)

                # update GMM on protocrl loss
                if global_step % protocrl_frequency == 0:                               
                    L_ProtoCRL = compute_ProtoCRL_Loss(data, global_step, actor_loss, posterior, log_posterior, log_likelihood, log_prior)
                    gmm_optimizer.zero_grad()
                    L_ProtoCRL.backward()
                    if global_step is not None and global_step % 50 == 0:
                        writer.add_scalar("Grads/mu_norm", actor.mu.grad.norm().item(), global_step)
                        writer.add_scalar("Grads/log_sigma_norm", actor.log_sigma.grad.norm().item(), global_step)
                        writer.add_scalar("Grads/pi_norm", actor.pi.grad.norm().item(), global_step)
                    torch.nn.utils.clip_grad_norm_(gmm_params, max_norm=1.0)
                    gmm_optimizer.step()

                torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
                actor_optimizer.step()
                                                                
                with torch.no_grad():
                    _, log_pi, *_ = actor.get_action(data.observations, global_step)
                if global_step % 50 == 0:
                    writer.add_scalar("SAC/alpha", alpha, global_step)
                    writer.add_scalar("SAC/log_pi_mean", log_pi.mean().item(), global_step)
                    writer.add_scalar("SAC/log_pi_max", log_pi.abs().max().item(), global_step)

                if global_step % target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                        
            state = next_state
            global_step += 1
            pbar.update(1)
            action, _, _, posterior, *_ = actor.get_action(torch.FloatTensor(state).unsqueeze(0).to(device), global_step)
            action = action[0].detach().cpu().numpy()

            if global_step % 500 == 0:
                fill = {f"table_{k}": (float(BUFFER_SIZE) if rb.full[k] else float(rb.pos[k])) for k in range(1, K+1)}
                writer.add_scalars("ReplayBuffer/table_fill", fill, global_step)

            if global_step % 500 == 0:
                maybe_revive(rb, global_step)

            if global_step % 2000 == 0:
                log_table_routing(rb, global_step)

            if global_step % 50000 == 0:
                # evaluate all tasks
                for t in range(min(env.cur_seq_idx + 1, NUM_TASKS)):
                    eval_single_task(t, global_step)

        episode_counter += 1
        writer.add_scalar("Environment/Episode_Reward", ep_reward, episode_counter)
        writer.add_scalar("Environment/Success_Rate", success, episode_counter)

    print("Training complete, running final evaluation on all tasks...")
    evaluate_past_tasks(NUM_TASKS - 1, global_step) 


# FUNCTION TO EVALUATE PERFORMANCE ON ALL TASKS
def evaluate_past_tasks(current_task_id, step, eval_episodes=100):
    for task_id in range(current_task_id + 1):
        task_name = CW10_V2[task_id]
        eval_env = cw_envs.get_single_env(
            task=task_name,
            one_hot_idx=task_id,
            one_hot_len=NUM_TASKS
        )
        successes = 0

        for _ in range(eval_episodes):
            state = eval_env.reset()
            done = False
            step_count = 0

            while not done and step_count < 201:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, mean_action, *_ = actor.get_action(state_tensor, step)
                    action = mean_action[0].cpu().numpy()

                state, reward, done, info = eval_env.step(action)
                step_count += 1
                if info.get("success", False):
                    successes += 1
                    done = True
                    break

        eval_env.close()
        success_rate = successes / eval_episodes
        performance_matrix[current_task_id, task_id] = success_rate
        writer.add_scalar(f"Evaluation/Task_{task_id}_Success", success_rate, step)

    # compute forgetting
    if current_task_id > 0:
        forgetting_list = []

        # compute forgetting for all past tasks
        for j in range(current_task_id): 
            max_past_performance = np.max(performance_matrix[:current_task_id, j])          # max performance on task j
            current_performance = performance_matrix[current_task_id, j]                    # current performance on task j
            forgetting_j = max_past_performance - current_performance
            forgetting_list.append(forgetting_j)
            
            writer.add_scalar(f"Evaluation/Task_{j}_Forgetting", forgetting_j, current_task_id)

        avg_forgetting = np.mean(forgetting_list)
        avg_success = np.mean(performance_matrix[current_task_id, :current_task_id + 1])
        
        writer.add_scalar("Evaluation/Average_Forgetting", avg_forgetting, current_task_id)
        writer.add_scalar("Evaluation/Average_Success", avg_success, current_task_id)
        
    return


def eval_single_task(task_id, step, num_episodes=40):
    task_name = CW10_V2[task_id]
    eval_env = cw_envs.get_single_env(task=task_name, one_hot_idx=task_id, one_hot_len=NUM_TASKS)
    successes = 0
    for _ in range(num_episodes):
        state = eval_env.reset()
        done, step_count = False, 0
        while not done and step_count < 201:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, mean_action, *_ = actor.get_action(state_tensor, step)
                action = mean_action[0].cpu().numpy()
            state, reward, done, info = eval_env.step(action)
            step_count += 1
            if info.get("success", False):
                successes += 1
                break
    eval_env.close()
    writer.add_scalar(f"Environment/Task{task_id}_Running_Success", successes / num_episodes, step)

@torch.no_grad()
# check wether obs which were saved in cluster k are still in cluster k 
def log_table_routing(rb, step):
    for k in range(1, K + 1):
        n = BUFFER_SIZE if rb.full[k] else int(rb.pos[k])
        if n < 256:
            continue
        idx = np.random.randint(0, n, size=512)
        obs = torch.as_tensor(rb.observations[k, idx], device=device)
        _, _, _, post, *_ = actor.get_action(obs, None)
        same = (post.argmax(dim=-1) == (k - 1)).float().mean().item()
        writer.add_scalar(f"Routing/table_{k}_still_cluster_{k}", same, step)

NOVELTY_MIN = 30.0        # abs value for a state to be considered novel
NOVELTY_JUMP = 1.75       # multplier, for a state to be considered novel it must be this times the mean
NOVEL_FRAC = 0.5          # at least this of the window must be novel
NOVEL_WINDOW = 500        # newest steps to look at
REVIVE_START = 50_000     # start watching after this many steps
REVIVE_COOLDOWN = 50_000  # minimum steps between two revivals
revive_state = {"base": None, "last": -10**9}

@torch.no_grad()
def revive_dead_cluster(obs_novel):
    k_new = int(F.softmax(actor.pi, dim=0).argmin())          # least-used cluster
    h = F.normalize(actor.encoder(obs_novel), dim=-1) * 20.0
    actor.mu.data[k_new] = h.mean(dim=0)                       # move it onto the novel states
    actor.pi.data[k_new] = actor.pi.data.max()                 # and remove its prior handicap
    return k_new

@torch.no_grad()
# checking if there is a new set of data to rerout it to a dead cluster
def maybe_revive(rb, step):
    if step < REVIVE_START:
        return
    idx = (rb.pos[0] - 1 - np.arange(NOVEL_WINDOW)) % BUFFER_SIZE       # newest rows of table 0
    obs = torch.as_tensor(rb.observations[0, idx], device=device)
    # we compute novelty as the max negative log-likelihood: basically we're checking if a new set of states appears which is further from the mean compared to the usual distance
    novelty = -actor.get_action(obs, None)[5].max(dim=-1).values        
    med = novelty.median().item()                                   # median distance (negative log likelihood) from the mean
    if revive_state["base"] is None:
        revive_state["base"] = med
    thresh = max(NOVELTY_MIN, NOVELTY_JUMP * revive_state["base"])
    novel = novelty > thresh
    frac = novel.float().mean().item()

    writer.add_scalar("Novelty/median", med, step)
    writer.add_scalar("Novelty/baseline", revive_state["base"], step)
    writer.add_scalar("Novelty/frac_novel", frac, step)
    if frac < NOVEL_FRAC:
        revive_state["base"] += 0.02 * (med - revive_state["base"])      # changing the baseline slowly
        return
    if step - revive_state["last"] < REVIVE_COOLDOWN:
        return
    if F.softmax(actor.pi, dim=0).min().item() > 0.05:                   # no dead cluster left
        return
    k_new = revive_dead_cluster(obs[novel])
    rb.pos[k_new + 1] = 0                                                # empty its table (old warm-up rows)
    rb.full[k_new + 1] = False
    revive_state["last"] = step
    writer.add_scalar("Novelty/revived_cluster", k_new + 1, step)
    tqdm.write(f"step {step}: {frac:.0%} of the last {NOVEL_WINDOW} states are novel -> cluster {k_new + 1} moved onto them")


ProtoCRL_SAC()