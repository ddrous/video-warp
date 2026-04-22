
#%%
import numpy as np
import gymnasium as gym
import minigrid
from minigrid.envs import * # Forces environment registration
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from minigrid.core.world_object import Goal

## Fix the seed for reproducibility
SEED = 42
np.random.seed(SEED)      

# ─────────────────────────────────────────────
#  Expert policies
# ─────────────────────────────────────────────

def get_expert_action_empty(env):
    """
    Heuristic expert for 'Empty' MiniGrid environments (no obstacles).
    """
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    goal_pos  = (env.unwrapped.width - 2, env.unwrapped.height - 2)

    if agent_pos[0] < goal_pos[0]:
        if agent_dir == 0:   return 2   # East  → forward
        elif agent_dir == 3: return 1   # North → turn right
        else:                return 0   # other → turn left

    elif agent_pos[1] < goal_pos[1]:
        if agent_dir == 1:   return 2   # South → forward
        elif agent_dir == 0: return 1   # East  → turn right
        else:                return 0   # other → turn left

    return 5  # already at goal / fallback


def get_expert_action_bfs(env):
    """
    Best-effort BFS expert.
    Finds the shortest path to the goal. If the goal is blocked, 
    it finds the path to the reachable cell closest to the goal.
    """
    unwrapped = env.unwrapped
    grid      = unwrapped.grid
    agent_pos = tuple(unwrapped.agent_pos)
    agent_dir = unwrapped.agent_dir

    # ── Find the goal cell ──────────────────────────────────────────────────
    goal_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == 'goal':
                goal_pos = (x, y)
                break
        if goal_pos:
            break

    if goal_pos is None:
        return 5  

    # Manhattan distance helper
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # ── BFS over (x, y, direction) ──────────────────────────────────────────
    DIR_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    start   = (agent_pos[0], agent_pos[1], agent_dir)
    queue   = deque([(start, [])])          
    visited = {start}

    # Track the closest we can possibly get
    closest_dist = manhattan(agent_pos, goal_pos)
    best_action  = 5

    while queue:
        (x, y, d), actions = queue.popleft()

        # Update the best known action if we found a cell closer to the goal
        dist = manhattan((x, y), goal_pos)
        if dist < closest_dist:
            closest_dist = dist
            best_action = actions[0] if actions else 5

        # If we found the actual goal, take the direct path immediately
        if (x, y) == goal_pos:
            return actions[0] if actions else 5

        # Action 0 – turn left
        nd = (d - 1) % 4
        s  = (x, y, nd)
        if s not in visited:
            visited.add(s)
            queue.append((s, actions + [0]))

        # Action 1 – turn right
        nd = (d + 1) % 4
        s  = (x, y, nd)
        if s not in visited:
            visited.add(s)
            queue.append((s, actions + [1]))

        # Action 2 – move forward
        dx, dy = DIR_VECS[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.width and 0 <= ny < grid.height:
            cell = grid.get(nx, ny)
            if cell is None or cell.can_overlap():
                s = (nx, ny, d)
                if s not in visited:
                    visited.add(s)
                    queue.append((s, actions + [2]))

    # If the queue empties and we didn't find the goal, return the move 
    # that gets us toward the closest reachable tile we found.
    return best_action


def get_expert_action_random(env):
    """
    Random exploration policy.
    Chooses randomly between turning left, turning right, and moving forward.
    If the agent miraculously lands on the goal, it stops (returns 5).
    """
    unwrapped = env.unwrapped
    agent_pos = unwrapped.agent_pos
    grid = unwrapped.grid

    # Check if the agent is currently standing on the goal
    current_cell = grid.get(agent_pos[0], agent_pos[1])
    if current_cell is not None and current_cell.type == 'goal':
        return 5  # We made it! Stop/Done.

    # Otherwise, explore randomly using only Left (0), Right (1), or Forward (2)
    return int(np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.5]))
    # return int(np.random.choice([0, 1, 2], p=[0.0, 0.0, 1.0]))



# def get_expert_action_random(env):
#     """
#     Smart random exploration policy.
#     Checks the cells in front, to the left, and to the right. 
#     It will only select actions that point toward open, walkable tiles.
#     If multiple paths are open, it samples based on the normalized base probabilities.
#     """
#     unwrapped = env.unwrapped
#     agent_pos = unwrapped.agent_pos
#     agent_dir = unwrapped.agent_dir
#     grid = unwrapped.grid

#     # 1. Check if the agent is currently standing on the goal
#     current_cell = grid.get(agent_pos[0], agent_pos[1])
#     if current_cell is not None and current_cell.type == 'goal':
#         return 5  # We made it! Stop/Done.

#     # MiniGrid Directions: 0: East, 1: South, 2: West, 3: North
#     DIR_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

#     def is_walkable(d):
#         """Helper to check if the adjacent cell in direction `d` is free."""
#         dx, dy = DIR_VECS[d]
#         nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
        
#         # Check grid boundaries
#         if 0 <= nx < grid.width and 0 <= ny < grid.height:
#             cell = grid.get(nx, ny)
#             # A cell is walkable if it's empty (None) or allows overlapping (e.g. open doors/goals)
#             if cell is None or cell.can_overlap():
#                 return True
#         return False

#     valid_actions = []
#     base_probs = []

#     # Action 0: Turn Left. (Is the tile to our relative left walkable?)
#     if is_walkable((agent_dir - 1) % 4):
#         valid_actions.append(0)
#         base_probs.append(0.25)

#     # Action 1: Turn Right. (Is the tile to our relative right walkable?)
#     if is_walkable((agent_dir + 1) % 4):
#         valid_actions.append(1)
#         base_probs.append(0.25)

#     # Action 2: Move Forward. (Is the tile directly in front walkable?)
#     if is_walkable(agent_dir):
#         valid_actions.append(2)
#         base_probs.append(0.50)

#     # 2. Dead-end Fallback
#     # If front, left, and right are ALL blocked, we are in a dead end. 
#     # We must allow the agent to turn left or right so it can eventually face backwards and escape.
#     if not valid_actions:
#         return int(np.random.choice([0, 1]))

#     # 3. Normalize the probabilities for the valid actions
#     # Example: If only Forward(0.5) and Left(0.25) are valid, they become 66.6% and 33.3%
#     probs = np.array(base_probs)
#     probs = probs / probs.sum()

#     return int(np.random.choice(valid_actions, p=probs))

# ─────────────────────────────────────────────
#  Video generation
# ─────────────────────────────────────────────

POLICY_MAP = {
    'empty': get_expert_action_empty,
    'bfs':   get_expert_action_bfs,
    'random': get_expert_action_random,
}

def generate_bc_videos(N=10, T=32,
                        env_id='MiniGrid-Empty-Random-5x5-v0',
                        policy='empty'):
    
    if policy not in POLICY_MAP:
        raise ValueError(f"Unknown policy '{policy}'")
    
    expert = POLICY_MAP[policy]
    env    = gym.make(env_id, render_mode='rgb_array')

    # obs, info = env.reset()     #$ TODO: put this inside the loop for new environments

    # 1. Force the output resolution to be 72x72 natively
    # A 9x9 grid with a tile_size of 8 pixels = exactly 72x72 image.
    env.unwrapped.tile_size = 8
    env.unwrapped.highlight = False 

    videos = np.zeros((N, T, 72, 72, 3), dtype=np.uint8)
    # videos = np.zeros((N, T, 88, 88, 3), dtype=np.uint8)
    # videos = np.zeros((N, T, 128, 128, 3), dtype=np.uint8)
    actions = np.full((N, T), 3, dtype=np.int32)

    ## Have a dictionaly that maps action 3 to 5, and vice versa. Better this way !
    action_dict = {0:0, 1:1, 2:2, 3:5, 4:4, 5:3, 6:6}

    for n in range(N):

        obs, info = env.reset(seed=SEED+n)     #$ TODO: put this outside the loop to reuse the environment

        # 1. Manually reset the step counter to prevent early truncation
        env.unwrapped.step_count = 0

        # 1. Find and remove the default goal
        grid = env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    grid.set(x, y, None) # Erase the old goal
        
        # 2. Place a new goal in a random empty tile
        env.unwrapped.place_obj(Goal())

        # 2. Randomize starting position and direction
        # We must unset the hardcoded (1, 1) first so the algorithm knows it's empty
        env.unwrapped.agent_pos = (-1, -1) 
        env.unwrapped.place_agent() # Natively places agent in a random empty tile with random dir

        frame = env.render()
        done  = False

        for t in range(T):
            videos[n, t] = frame

            if not done:
                action = expert(env)
                actions[n, t] = action_dict[action]
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                done  = terminated or truncated

    env.close()
    return videos, actions


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def visualize_video(videos, video_idx=0, interval=150, title="MiniGrid episode"):
    frames = videos[video_idx]   
    T      = len(frames)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle(f"{title}  –  episode {video_idx}", fontsize=10)
    img = ax.imshow(frames[0])
    ax.axis('off')
    step_text = ax.set_title(f"t = 0 / {T-1}", fontsize=9)

    def update(t):
        img.set_data(frames[t])
        step_text.set_text(f"t = {t} / {T-1}")
        return img, step_text

    ani = animation.FuncAnimation(
        fig, update,
        frames=T,
        interval=interval,
        blit=True,
        repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return ani   


# def visualise_frames(videos, video_idx=0, interval=150, title="MiniGrid episode"):
#     frames = videos[video_idx]   
#     T      = len(frames)

#     cols = int(np.ceil(np.sqrt(T)))
#     rows = int(np.ceil(T / cols))

#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
#     fig.suptitle(f"{title}  –  episode {video_idx}", fontsize=10)

#     axes_flat = np.array(axes).flatten()

#     for t, ax in enumerate(axes_flat):
#         if t < T:
#             ax.imshow(frames[t])
#             ax.set_title(f"t={t}", fontsize=6)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

def visualise_frames(videos, actions, video_idx=0, interval=150, title="MiniGrid episode"):
    # Map Minigrid action integers to readable strings for the plot titles
    ACTION_NAMES = {
        0: "Left",
        1: "Right",
        2: "Forward",
        3: "Toggle/Done",   # Note: The original 5 is 'Toggle' natively, but we prefer 3, and our expert uses it as a fallback/done
        4: "Drop",          ## Not used
        5: "Pickup",        ## Not used
        6: "Done"           ## Not used
    }

    frames = videos[video_idx]
    episode_actions = actions[video_idx] # Get the specific actions for this video
    T = len(frames)

    cols = int(np.ceil(np.sqrt(T)))
    rows = int(np.ceil(T / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"{title}  –  episode {video_idx}", fontsize=10)

    # Flatten axes for easy iteration (handles single-frame edge cases safely)
    axes_flat = np.array(axes).flatten()

    for t, ax in enumerate(axes_flat):
        if t < T:
            ax.imshow(frames[t])
            
            # Fetch the action and its string representation
            act_int = episode_actions[t]
            act_str = ACTION_NAMES.get(act_int, str(act_int))
            
            # Update the title to show the time step and the action taken
            ax.set_title(f"t={t} | {act_str}", fontsize=8)
            
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # num_videos       = 500*16
    num_videos       = int(1000)
    # num_videos       = int(10)
    temporal_horizon = 10          

    env_id_empty  = 'MiniGrid-Empty-16x16-v0'
    # env_id_obstacle = 'MiniGrid-SimpleCrossingS11N5-v0'
    env_id_obstacle = 'MiniGrid-SimpleCrossingS9N1-v0'

    CHOSEN_ENV    = env_id_obstacle   

    # CHOSEN_POLICY = 'bfs'       ## Could be 'empty' or 'random'
    CHOSEN_POLICY = 'random'

    print(f"Env    : {CHOSEN_ENV}")
    print(f"Policy : {CHOSEN_POLICY}")
    print(f"Generating {num_videos} videos (T={temporal_horizon})…")

    dataset, actions = generate_bc_videos(
        N      = num_videos,
        T      = temporal_horizon,
        env_id = CHOSEN_ENV,
        policy = CHOSEN_POLICY,
    )

    print(f"Done!  dataset shape={dataset.shape} optional action shape={actions.shape}  dtype={dataset.dtype}")

    # Saves perfectly at (8000, 16, 72, 72, 3) without the ::4 downsampling
    # np.save('minigrid.npy', dataset) 
    np.savez('minigrid_labelled.npz', dataset=dataset, actions=actions)

    visualise_frames(dataset, actions, video_idx=0, title=f"{CHOSEN_ENV}  –  all frames")

#%%
id_to_plot = np.random.randint(num_videos)
visualise_frames(dataset, actions, video_idx=id_to_plot, title=f"{CHOSEN_ENV}  –  {id_to_plot}")


## Print all the unique ids in the dataset to verify diversity
np.unique(actions)

# print(actions)