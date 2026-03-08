import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import spaces

class AquaBotEnv(gym.Env):
    """
    AquaBot River Cleaning Environment
    - River channel with banks (walls)
    - Floating debris targets to collect
    - River current disturbance
    - Obstacle avoidance (logs, rocks)
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.river_length = 20.0
        self.river_width = 6.0
        
        self.robot_pos = np.array([1.0, 3.0])
        self.robot_angle = 0.0
        self.robot_speed = 0.3
        
        self.current_strength = 0.05
        
        self.num_debris = 8
        self.debris = []
        self.obstacles = []
        
        self.grid_cols = 10
        self.grid_rows = 3
        self.coverage_grid = np.zeros((self.grid_rows, self.grid_cols))
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # 0=forward, 1=turn left, 2=turn right, 3=collect
        self.action_space = spaces.Discrete(4)
        
        self.max_steps = 300
        self.step_count = 0
        self.total_collected = 0
        self.render_mode = render_mode
        self.fig = None

    def _generate_debris(self):
        debris = []
        for _ in range(self.num_debris):
            x = np.random.uniform(2.0, self.river_length - 1.0)
            y = np.random.uniform(0.8, self.river_width - 0.8)
            debris.append([x, y, False])
        return debris

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(3):
            x = np.random.uniform(3.0, self.river_length - 2.0)
            y = np.random.uniform(1.0, self.river_width - 1.0)
            obstacles.append([x, y])
        return obstacles

    def _get_obs(self):
        rx, ry = self.robot_pos
        
        dist_left  = ry / self.river_width
        dist_right = (self.river_width - ry) / self.river_width
        
        uncollected = [[d[0], d[1]] for d in self.debris if not d[2]]
        if uncollected:
            dists = [np.hypot(d[0]-rx, d[1]-ry) for d in uncollected]
            nearest = uncollected[np.argmin(dists)]
            dx = (nearest[0] - rx) / self.river_length
            dy = (nearest[1] - ry) / self.river_width
            dist = min(dists) / self.river_length
        else:
            dx, dy, dist = 0.0, 0.0, 1.0
        
        if self.obstacles:
            obs_dists = [np.hypot(o[0]-rx, o[1]-ry) for o in self.obstacles]
            nearest_obs = min(obs_dists) / self.river_length
        else:
            nearest_obs = 1.0
        
        coverage_pct = self.coverage_grid.mean()
        norm_x     = rx / self.river_length
        norm_y     = ry / self.river_width
        norm_angle = self.robot_angle / np.pi
        
        obs = np.array([
            norm_x, norm_y, norm_angle,
            dist_left, dist_right,
            dx, dy, dist,
            nearest_obs, coverage_pct
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)

    def _update_coverage(self):
        rx, ry = self.robot_pos
        col = int(rx / self.river_length * self.grid_cols)
        row = int(ry / self.river_width  * self.grid_rows)
        col = np.clip(col, 0, self.grid_cols - 1)
        row = np.clip(row, 0, self.grid_rows - 1)
        self.coverage_grid[row][col] = 1.0

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        
        # River current
        self.robot_pos[1] -= self.current_strength
        
        # Movement
        if action == 0:
            self.robot_pos[0] += self.robot_speed * np.cos(self.robot_angle)
            self.robot_pos[1] += self.robot_speed * np.sin(self.robot_angle)
        elif action == 1:
            self.robot_angle += 0.2
        elif action == 2:
            self.robot_angle -= 0.2
        # action == 3: collect attempt
        
        self.robot_angle = self.robot_angle % (2 * np.pi)
        self._update_coverage()
        
        rx, ry = self.robot_pos
        terminated = False
        
        # ---- REWARD FUNCTION v2 ----

        # 1. Coverage reward
        reward += self.coverage_grid.mean() * 0.3

        # 2. Debris rewards
        uncollected = [d for d in self.debris if not d[2]]
        if uncollected:
            dists = [np.hypot(d[0]-rx, d[1]-ry) for d in uncollected]
            min_dist = min(dists)

            # Pull robot toward nearest debris
            reward += (1.0 / (min_dist + 0.1)) * 0.15

            # Collection check
            for d in uncollected:
                dist_to_debris = np.hypot(d[0]-rx, d[1]-ry)
                if dist_to_debris < 0.5:
                    if action == 3:
                        d[2] = True
                        self.total_collected += 1
                        reward += 15.0
                    else:
                        reward += 0.5

        # 3. Bank penalty
        if ry < 0.3 or ry > self.river_width - 0.3:
            reward -= 8.0
            terminated = True

        # 4. Obstacle penalty
        for obs in self.obstacles:
            if np.hypot(obs[0]-rx, obs[1]-ry) < 0.4:
                reward -= 8.0
                terminated = True

        # 5. Out of bounds
        if rx < 0 or rx > self.river_length:
            reward -= 3.0
            terminated = True

        # 6. Small step penalty
        reward -= 0.005

        # 7. All debris collected bonus
        if all(d[2] for d in self.debris):
            reward += 100.0
            terminated = True

        # ---------------------------------
        
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        
        info = {
            'collected': self.total_collected,
            'coverage': self.coverage_grid.mean()
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos  = np.array([1.0, self.river_width / 2])
        self.robot_angle = 0.0
        self.step_count = 0
        self.total_collected = 0
        self.coverage_grid = np.zeros((self.grid_rows, self.grid_cols))
        self.debris    = self._generate_debris()
        self.obstacles = self._generate_obstacles()
        return self._get_obs(), {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 4))
        
        self.ax.clear()
        
        river = patches.Rectangle((0, 0), self.river_length,
                                   self.river_width, color='lightblue')
        self.ax.add_patch(river)
        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.coverage_grid[r][c]:
                    cx = c / self.grid_cols * self.river_length
                    cy = r / self.grid_rows * self.river_width
                    cell = patches.Rectangle(
                        (cx, cy),
                        self.river_length / self.grid_cols,
                        self.river_width / self.grid_rows,
                        color='lightgreen', alpha=0.4
                    )
                    self.ax.add_patch(cell)
        
        for obs in self.obstacles:
            circle = plt.Circle(obs, 0.3, color='brown')
            self.ax.add_patch(circle)
        
        for d in self.debris:
            color = 'gray' if d[2] else 'orange'
            circle = plt.Circle((d[0], d[1]), 0.2, color=color)
            self.ax.add_patch(circle)
        
        rx, ry = self.robot_pos
        arrow_dx = 0.5 * np.cos(self.robot_angle)
        arrow_dy = 0.5 * np.sin(self.robot_angle)
        self.ax.annotate('', xy=(rx + arrow_dx, ry + arrow_dy),
                         xytext=(rx, ry),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))
        robot_circle = plt.Circle((rx, ry), 0.25, color='red')
        self.ax.add_patch(robot_circle)
        
        self.ax.set_xlim(0, self.river_length)
        self.ax.set_ylim(-0.5, self.river_width + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(
            f"AquaBot | Debris: {self.total_collected}/{self.num_debris} | "
            f"Coverage: {self.coverage_grid.mean()*100:.0f}%"
        )
        plt.pause(0.01)

    def close(self):
        if self.fig:
            plt.close(self.fig)