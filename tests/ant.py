import gymnasium 
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset
from gymnasium .wrappers import RecordVideo
import torch.nn.functional as F
import copy
from collections import deque
import random

class ReplayBuffer:
    """経験を保存するためのリプレイバッファ"""
    def __init__(self, capacity):
        # デックで固定長のバッファを初期化
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """バッファに経験を追加"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """バッファからランダムに経験をサンプリング"""
        return random.sample(self.buffer, batch_size)

class GymDataset(IterableDataset):
    """リプレイバッファからのサンプリング結果を使用してPyTorchデータセットを作成"""
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        # リプレイバッファからサンプルを取得し、テンソルに変換してイテレータとして返す
        for experience in self.buffer.sample(self.sample_size):
            tensors = [torch.tensor(item, dtype=torch.float32) if not isinstance(item, float) else torch.tensor([item], dtype=torch.float32) for item in experience]
            yield tuple(tensors)

def init_weight(layer, initializer="he normal"):
    """ネットワークの重みの初期化を行う関数"""
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Critic(nn.Module):
    """クリティックネットワークの定義"""
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(Critic, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        # ネットワークの層の定義
        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        """フォワードパスの定義"""
        x = torch.cat([states, actions], dim=1)  # 状態とアクションを結合
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)

class Actor(nn.Module):
    """アクターネットワークの定義。このネットワークは環境の状態を取り、行動の確率分布を出力する。"""

    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds  # 行動の上限・下限

        # ネットワークの層の定義
        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)  # 重みの初期化
        self.hidden1.bias.data.zero_()

        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        # 各行動の平均を出力するための層
        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        # 各行動の標準偏差の対数を出力するための層
        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        """フォワードパスの定義。確率分布を返す。"""
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        # 標準偏差を取得（-20から2の範囲でクリップ）
        std = log_std.clamp(min=-20, max=2).exp()
        # 正規分布を返す
        dist = Normal(mu, std)
        return dist

    def sample(self, states):
        """アクションをサンプルし、そのアクションの対数確率を返す。"""
        dist = self(states)
        # Reparameterization trickを使用してアクションをサンプル
        u = dist.rsample()
        action = torch.tanh(u)
        # アクションの確率を計算
        log_prob = dist.log_prob(u)
        # アクションが-1から1の範囲になるように修正
        log_prob -= torch.log((1 - action ** 2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        # 行動の上限をかけることで行動の範囲を調整
        return action * self.action_bounds[1], log_prob

class SACAgent(pl.LightningModule):
    def __init__(self, env, max_episodes=2000, batch_size=4096, gamma=0.99, tau=0.005, lr=1e-3, capacity=1000000, alpha=0.2, samples_per_epoch=256, learn_start_size=10000):
        super(SACAgent, self).__init__()
        # 環境の情報を取得
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_scale = [env.action_space.low[0], env.action_space.high[0]]

        # SACアルゴリズムのハイパーパラメータを設定
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.max_episodes = max_episodes
        self.capacity = capacity
        self.learn_start_size = learn_start_size
        self.alpha = alpha
        self.samples_per_epoch = samples_per_epoch

        # 報酬を保持するデックの初期化
        self.episode_rewards = deque(maxlen=50)

        # モデルの初期化
        self.actor = Actor(self.state_dim, self.action_dim, self.action_scale).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # リプレイバッファの初期化
        self.replay_buffer = ReplayBuffer(capacity)
        
        self.episode_count = 0

        # PyTorch Lightningの自動最適化を無効にする
        self.automatic_optimization = False

        # 環境の初期状態を取得
        self.current_state, _ = self.env.reset()
        self.current_reward = 0
        self.current_episode = 0
        self.current_step = 0
        self.max_step = 1000

    def prepare_replay_buffer(self):
        """リプレイバッファを準備する"""
        state, _ = self.env.reset()
        while len(self.replay_buffer) < self.batch_size:
            action = self.env.action_space.sample()
            next_state, reward, done, _, info = self.env.step(action)
            self.append_to_replay_buffer(state, action, reward, next_state, done)
            if not done:
                state = next_state
            else:
                state, _ = self.env.reset()

    def train_dataloader(self):
        """訓練データローダーを返す"""
        # データセットの作成
        dataset = GymDataset(self.replay_buffer, self.samples_per_epoch)
        # データローダーの作成
        dataloader = DataLoader(dataset, self.batch_size)
        return dataloader

    def soft_update(self, target_network, local_network, tau=0.005):
        """ソフトアップデートを行う"""
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def train_step(self, batch):
        """1ステップの訓練を行う"""
        states, actions, rewards, next_states, dones = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # Q値のターゲットを計算
        with torch.no_grad():
            next_actions, target_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            next_action_values = torch.min(target_q1, target_q2)
            v = (1 - dones) * (next_action_values - self.alpha * target_log_probs)
            target_values = rewards + self.gamma * v

        # 現在のQ値を取得
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        # クリティックの損失を計算
        critic_loss1 = F.mse_loss(q1, target_values)
        critic_loss2 = F.mse_loss(q2, target_values)

        # アクターの損失を計算
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        return critic_loss1, critic_loss2, actor_loss

    @torch.no_grad()
    def play_episode(self):
        """エピソードの実行を行い、リプレイバッファに結果を追加する"""

        # 現在のステートからアクターネットワークを使って行動をサンプルする
        states = np.expand_dims(self.current_state, axis=0)
        state_tensor = torch.from_numpy(states).float().to(self.device)
        action, _ = self.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]

        # 選択された行動を環境に適用し、新しいステートと報酬を取得する
        next_state, reward, done, _, info = self.env.step(action)

        # 経験をリプレイバッファに保存
        self.append_to_replay_buffer(self.current_state, action, reward, next_state, done)

        # 状態、報酬、ステップ数を更新
        self.current_state = next_state
        self.current_reward += reward
        self.current_step += 1

        # エピソードが終了した場合、またはステップ数が最大値を超えた場合の処理
        if done or self.current_step > self.max_step:
            # エピソード数をインクリメント
            self.current_episode += 1

            # ログ情報を保存
            self.log('train/episode_num', torch.tensor(self.current_episode, dtype=torch.float32))
            self.log('train/episode_reward', self.current_reward)
            self.log('train/current_step', self.current_step)

            # 過去の報酬の平均値を計算
            self.episode_rewards.append(self.current_reward)
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.log('train/avg_episode_reward', avg_reward)

            # 新しいエピソードのための状態をリセット
            self.current_state, _ = self.env.reset()

            # 100エピソードごとの情報をプリント
            if self.current_episode % 100 == 0:
                print(f"episode:{self.current_episode}, episode_reward: {self.current_reward}, avg_reward: {avg_reward}")

            # 現在の報酬とステップ数をリセット
            self.current_reward = 0
            self.current_step = 0

            # エピソード数が最大値に達した場合、トレーニングを終了
            if self.current_episode >= self.max_episodes:
                self.trainer.should_stop = True

        return done

    def training_step(self, batch, batch_idx):
        """訓練ステップを実行する"""
        current_episode_done = self.play_episode()
        states, actions, rewards,  next_states, dones = map(torch.squeeze, batch)

        states, actions, rewards, next_states, dones = states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device)

        critic_loss1, critic_loss2, actor_loss = self.train_step((states, actions, rewards, next_states, dones))
        
        opt_actor, opt_critic1, opt_critic2  = self.optimizers()
    
        # クリティック1の最適化
        opt_critic1.zero_grad()
        self.manual_backward(critic_loss1)
        opt_critic1.step()
        
        # クリティック2の最適化
        opt_critic2.zero_grad()
        self.manual_backward(critic_loss2)
        opt_critic2.step()

        # アクターの最適化
        opt_actor.zero_grad()
        self.manual_backward(actor_loss)
        opt_actor.step()

        self.soft_update(self.target_critic1, self.critic1, self.tau)
        self.target_critic1.eval()
        self.soft_update(self.target_critic2, self.critic2, self.tau)
        self.target_critic2.eval()

        if current_episode_done:
            self.log('train/critic_loss1', critic_loss1)
            self.log('train/critic_loss2', critic_loss2)
            self.log('train/actor_loss', actor_loss)
            self.log('train/alpha', self.alpha)
        
    def configure_optimizers(self):
        """オプティマイザを設定する"""
        critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=self.lr)
        critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=self.lr)
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        return actor_optimizer, critic_optimizer1, critic_optimizer2

    def forward(self, state):
        """アクターの方策を使用してアクションを生成する"""
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().numpy()
        
    def append_to_replay_buffer(self, state, action, reward, next_state, done):
        """リプレイバッファに結果を追加する"""
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

def setup_display():
    # Docker上なので仮想ディスプレイを作成する。
    from pyvirtualdisplay import Display
    Display(visible=False, size=(1400, 900)).start()

def main():
    # 仮想ディスプレイのセットアップ
    # setup_display()
    
    # OpenAI GymのAnt-v4環境を作成
    env = gymnasium.make('Ant-v4', render_mode="rgb_array")
    
    # 10エピソードごとにビデオを記録するように環境をラップ
    # env = RecordVideo(env, './videos', episode_trigger=lambda x: x % 10 == 0)

    # SACAgentモデルのインスタンス化
    model = SACAgent(env, lr=3e-4, max_episodes=5000, gamma=0.99,
                     batch_size=256, samples_per_epoch=256)
    
    # リプレイバッファの準備
    model.prepare_replay_buffer()
    
    # PyTorch Lightningトレーナのインスタンス化
    trainer = pl.Trainer(max_epochs=-1, accelerator="auto", 
                         log_every_n_steps=1, enable_progress_bar=False)
    
    # モデルのトレーニング開始
    trainer.fit(model)

if __name__ == "__main__":
    # スクリプトが直接実行された場合、main関数を呼び出す
    main()
