from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gymnasium
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# パラメータの設定
gym_game_name = 'Pendulum-v1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)
result_dir_path = Path('result')
model_dir_path = Path('model')
if not result_dir_path.exists():
    result_dir_path.mkdir(parents=True)
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)
# モデルの構築
class CriticNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):

        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class ActorNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, action_scale):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, output_dim)
        self.log_std_linear = nn.Linear(hidden_dim, output_dim)

        # デバイスは後で to(device) 呼び出しで揃える
        self.register_buffer('action_scale', torch.tensor(action_scale))
        self.register_buffer('action_bias', torch.tensor(0.0))

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        # state をネットワークのデバイスに揃える
        device = next(self.parameters()).device
        state = state.to(device)

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, mean

        return action, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
class ActorCriticModel(object):

    def __init__(self, state_dim, action_dim, action_scale, args, device):
        # カウンタ類
        self.total_it = 0
        # args にあわせる
        self.start_steps = args.get('start_steps', 1000)

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.target_update_interval = args['target_update_interval']
        self.device = device

        # 必要な次元・スケールを保持
        self.action_dim = action_dim
        self.action_scale_value = action_scale

        self.actor_net = ActorNet(input_dim=state_dim, output_dim=action_dim,
                                  hidden_dim=args['hiden_dim'], action_scale=action_scale).to(self.device)
        self.critic_net = CriticNet(input_dim=state_dim + action_dim, output_dim=1,
                                    hidden_dim=args['hiden_dim']).to(self.device)
        self.critic_net_target = CriticNet(input_dim=state_dim + action_dim, output_dim=1,
                                           hidden_dim=args['hiden_dim']).to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def select_action(self, state, evaluate=False):
        """
        state: ndarray (obs)
        evaluate: True -> 決定論的（mean）で返す
                  False -> サンプリングして返す（探索）
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # ランダム初期ステップ（CPUのnumpyから生成して env に渡す）
        if (not evaluate) and (self.total_it < self.start_steps):
            # env.action_space.sample() と同等の形にする (1D numpy)
            a = np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
            # total_it はインクリメントしておく
            self.total_it += 1
            return a

        # policy による行動
        with torch.no_grad():
            action_t, mean_t = self.actor_net.sample(state_t)   # tensors on device
            # actor_net.sample の返りは (action, mean)
            if evaluate:
                out = mean_t      # 決定論的に mean を使う
            else:
                out = action_t    # サンプリングされた action を使う

        # 学習時のみ total_it を更新（evaluate の場合は更新しない）
        if not evaluate:
            self.total_it += 1

        # GPU -> CPU -> numpy、1次元にして返す
        return out.squeeze(0).cpu().numpy().reshape(-1)



    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_state_action, _ = self.actor_net.sample(next_state_batch)
            next_q_values_target = self.critic_net_target(next_state_batch, next_state_action)
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q_values = self.critic_net(state_batch, action_batch)
        critic_loss = F.mse_loss(q_values, next_q_values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, _ = self.actor_net.sample(state_batch)
        q_values = self.critic_net(state_batch, action)
        actor_loss = - q_values.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)

        return critic_loss.item(), actor_loss.item()
def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target_net, source_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)


def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False
# メモリの構築
class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        # state/next_state を ndarray に、安全に変換して格納
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = float(reward)
        mask = float(mask)

        entry = (state, action, reward, next_state, mask)

        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = entry
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 各要素を ndarray に揃えて返す
        states = np.stack([b[0] for b in batch], axis=0)
        actions = np.stack([b[1] for b in batch], axis=0)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch], axis=0)
        masks = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, masks

    def __len__(self):
        return len(self.buffer)

# 連続値制御のActorCritiモデルの学習
args = {
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    'seed': 123456,
    'batch_size': 256,
    'hiden_dim': 256,
    'start_steps': 1000,
    'target_update_interval': 1,
    'memory_size': 100000,
    'epochs': 100,
    'eval_interval': 10
}
env = gymnasium.make(gym_game_name)
agent = ActorCriticModel(
    state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
    action_scale=env.action_space.high[0], args=args, device=device
)
memory = ReplayMemory(args['memory_size'])

episode_reward_list = []
eval_reward_list = []

n_steps = 0
n_update = 0
for i_episode in range(1, args['epochs'] + 1):

# episode 開始
    episode_reward = 0
    state, _ = env.reset()   # Gymnasium: (obs, info)
    done = False

    while not done:

        # action の形を必ず numpy 1D にする
        if args['start_steps'] > n_steps:
            action = env.action_space.sample()
            action = np.asarray(action, dtype=np.float32)
        else:
            action = agent.select_action(state, evaluate=False)   # 学習中は evaluate=False

        # バッチサイズに達していたら学習
        if len(memory) >= args['batch_size']:
            agent.update_parameters(memory, args['batch_size'], n_update)
            n_update += 1

        # Gymnasium の戻り値（obs, reward, terminated, truncated, info）
        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        n_steps += 1
        episode_reward += float(reward)

        # mask は episode 続行フラグ（終端なら0）
        memory.push(state=state, action=action, reward=reward, next_state=next_state, mask=float(not done))

        state = next_state

    episode_reward_list.append(episode_reward)

    if i_episode % args['eval_interval'] == 0:
# 評価（例）
        avg_reward = 0.0
        for _ in range(args['eval_interval']):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.select_action(state, evaluate=True)   # 評価は決定論的
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                state = next_state
            avg_reward += episode_reward
        avg_reward /= args['eval_interval']
        eval_reward_list.append(avg_reward)

        print("Episode: {}, Eval Avg. Reward: {:.0f}".format(i_episode, avg_reward))

print('Game Done !! Max Reward: {:.2f}'.format(np.max(eval_reward_list)))

torch.save(agent.actor_net.to('cpu').state_dict(), model_dir_path.joinpath(f'{gym_game_name}_actor.pth'))

plt.figure(figsize=(8, 6), facecolor='white')
g = sns.lineplot(
    data=pd.DataFrame({
        'episode': range(args['eval_interval'], args['eval_interval'] * (len(eval_reward_list) + 1), args['eval_interval']),
        'reward': eval_reward_list
    }),
    x='episode', y='reward', lw=2
)
plt.title('{}エピソードごとの学習済みモデルにおける\n評価報酬の平均値の推移'.format(args['eval_interval']), fontsize=18, weight='bold')
plt.xlabel('エピソード')
plt.ylabel('獲得報酬の平均値')
for tick in plt.yticks()[0]:
    plt.axhline(tick, color='grey', alpha=0.1)
plt.tight_layout()
plt.savefig(result_dir_path.joinpath('{}_eval_reward_{}.png'.format(gym_game_name, args['eval_interval'])), dpi=500)

print("check1")

state,_ = env.reset()
episode_reward = 0
done = False
while not done:
    with torch.no_grad():
        action = agent.select_action(state, evaluate=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    env.render()
    episode_reward += reward
    state = next_state
print('Reward: {:.2f}'.format(episode_reward))

result = []
for experiment_name in ['agent', 'random']:
    for i in tqdm(range(100)):

        # 修正版
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            state = next_state

        result.append([experiment_name, i, episode_reward])
result = pd.DataFrame(result, columns=['experiment_name', 'i', 'reward'])

g = sns.catplot(data=result, x='experiment_name', y='reward', kind='boxen')
g.fig.suptitle('学習済みモデルとランダムの報酬の比較', fontsize=18, weight='bold', y=1.0)
g.fig.set_figwidth(8)
g.fig.set_figheight(6)
g.fig.set_facecolor('white')
g.set_xlabels('')
g.set_ylabels('')
g.set_xticklabels(fontsize=16)
g.tight_layout()
g.savefig(result_dir_path.joinpath(f'{gym_game_name}_reward_agent_vs_random.png'), dpi=500)

from moviepy.editor import *
import warnings
warnings.filterwarnings('ignore')

from gymnasium.wrappers import RecordVideo

video_dir_path = Path('video')
video_dir_path.mkdir(exist_ok=True)

# RecordVideo 環境の作成
video_env = RecordVideo(
    gymnasium.make(gym_game_name, render_mode="rgb_array"),
    video_folder=str(video_dir_path),
    name_prefix=gym_game_name
)

state, _ = video_env.reset()
done = False
while not done:
    with torch.no_grad():
        action = agent.select_action(state, evaluate=True)
    next_state, reward, terminated, truncated, info = video_env.step(action)
    done = bool(terminated or truncated)
    state = next_state

video_env.close()


input_file_path = video_dir_path.joinpath(f'{gym_game_name}-episode-0.mp4')
output_file_path = video_dir_path.joinpath(f'{gym_game_name}-actor-critic.gif')
 
from moviepy.editor import VideoFileClip

output_file_path = video_dir_path.joinpath(f'{gym_game_name}-actor-critic.gif')

# 生成された動画ファイルを自動取得
video_files = list(video_dir_path.glob('*.mp4'))
if len(video_files) == 0:
    raise FileNotFoundError("動画ファイルが見つかりません")
input_file_path = video_files[0]

clip = VideoFileClip(str(input_file_path))
clip = clip.resize(width=600)
clip.write_gif(str(output_file_path), fps=14)
clip.close()

print("GIF生成完了:", output_file_path)
