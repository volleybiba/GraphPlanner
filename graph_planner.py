import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors


class FBGraphBuilder:
    #Строит граф по состояниям из offline-датасета

    def __init__(self, agent, dataset_observations, batch_size=2048):
        self.agent = agent
        self.observations = dataset_observations
        self.n_nodes = len(dataset_observations)

        self._jit_backward = jax.jit(
            lambda obs: self.agent.network.select('backward_repr')(obs)
        )

        self.embeddings = self._compute_embeddings(batch_size)
        print(f"Calculated {self.n_nodes} B-embeddings,"
              f"Dim: {self.embeddings.shape[1]}")

        self.graph = None
        self.knn_model = None

    def _compute_embeddings(self, batch_size):
        all_embeddings = []
        for start in range(0, self.n_nodes, batch_size):
            end = min(start + batch_size, self.n_nodes)
            batch = jnp.array(self.observations[start:end])
            emb = np.asarray(self._jit_backward(batch))
            all_embeddings.append(emb)
        return np.concatenate(all_embeddings, axis=0)

    def build_graph(self, k=10, metric='cosine'):
        self.knn_model = NearestNeighbors(
            n_neighbors=k, metric=metric, n_jobs=-1
        )
        self.knn_model.fit(self.embeddings)

        distances, indices = self.knn_model.kneighbors(self.embeddings)

        rows = np.repeat(np.arange(self.n_nodes), k)
        cols = indices.flatten()

        if metric == 'cosine':
            weights = 1.0 - distances.flatten() / 2.0
        else:
            weights = 1.0 / (1.0 + distances.flatten())

        mask = rows != cols
        rows, cols, weights = rows[mask], cols[mask], weights[mask]

        self.graph = csr_matrix(
            (weights, (rows, cols)),
            shape=(self.n_nodes, self.n_nodes)
        )

        # Проверка связности
        n_comp, labels = connected_components(self.graph, directed=False)
        print(f"Graph: {self.n_nodes} nodes, {self.graph.nnz} edges, "
              f"components: {n_comp}")
        if n_comp > 1:
            sizes = np.bincount(labels)
            print(f"5 largest components: {sorted(sizes, reverse=True)[:5]}")

        return self

    def find_nearest_node(self, observation):
        obs = jnp.array(observation[None])
        emb = np.asarray(self._jit_backward(obs))
        _, idx = self.knn_model.kneighbors(emb, n_neighbors=1)
        return int(idx[0, 0])

    def find_path(self, start_idx, goal_idx):
        dist_matrix, predecessors = dijkstra(
            self.graph, directed=False, indices=start_idx,
            return_predecessors=True
        )

        if np.isinf(dist_matrix[goal_idx]):
            return [start_idx, goal_idx]

        path = [goal_idx]
        current = goal_idx
        while current != start_idx:
            current = predecessors[current]
            if current < 0:
                break
            path.append(current)
        path.reverse()
        return path


class GraphPlanner:
    # Планировщик последовательности подцелей через граф

    def __init__(self, graph_builder, agent,
                 steps_per_subgoal=25, max_subgoals=10):
        self.graph_builder = graph_builder
        self.agent = agent
        self.steps_per_subgoal = steps_per_subgoal
        self.max_subgoals = max_subgoals

        self.current_plan = []
        self.current_subgoal_idx = 0
        self.plan_active = False
        self._steps_since_switch = 0

    def normalize_z(self, z):
        norm = jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8
        return z / norm * jnp.sqrt(self.agent.config['latent_dim'])

    def plan(self, observation, z_goal):
        # Строит план от текущего наблюдения к цели
        start_idx = self.graph_builder.find_nearest_node(observation)

        # Находим узел, ближайший к z_goal по косинусному сходству
        goal_emb = np.asarray(z_goal)[None]
        norms = np.linalg.norm(self.graph_builder.embeddings, axis=1,
                               keepdims=True) + 1e-8
        normed_emb = self.graph_builder.embeddings / norms
        normed_goal = goal_emb / (np.linalg.norm(goal_emb) + 1e-8)
        similarities = (normed_emb @ normed_goal.T).flatten()
        goal_idx = int(np.argmax(similarities))

        path_indices = self.graph_builder.find_path(start_idx, goal_idx)

        if len(path_indices) > self.max_subgoals:
            step = len(path_indices) / self.max_subgoals
            sampled = [path_indices[int(i * step)]
                       for i in range(self.max_subgoals)]
            sampled.append(path_indices[-1])
            path_indices = sampled

        # Пропуск первой подцели, т.к. она совпадает с текущим состоянием
        if len(path_indices) > 1:
            path_indices = path_indices[1:]

        subgoals = [self.graph_builder.embeddings[idx] for idx in path_indices]

        self.current_plan = subgoals
        self.current_subgoal_idx = 0
        self.plan_active = True
        self._steps_since_switch = 0

        print(f"Plan: {len(subgoals)} subgoals "
              f"(start: {start_idx}, goal: {goal_idx})")
        return subgoals

    def get_current_intention(self, observation):
        # Возвращает текущую интенцию
        if not self.plan_active or len(self.current_plan) == 0:
            raise RuntimeError("План не построен.")

        # Переключение по фиксированному числу шагов
        if (self._steps_since_switch >= self.steps_per_subgoal
                and self.current_subgoal_idx < len(self.current_plan) - 1):
            self.current_subgoal_idx += 1
            self._steps_since_switch = 0
            print(f"Subgoal {self.current_subgoal_idx + 1}/"
                  f"{len(self.current_plan)}")

        self._steps_since_switch += 1

        current_subgoal = self.current_plan[self.current_subgoal_idx]
        z = jnp.array(current_subgoal)
        z = self.normalize_z(z)
        return z

    def reset(self):
        self.current_plan = []
        self.current_subgoal_idx = 0
        self.plan_active = False
        self._steps_since_switch = 0


class GraphPlannerAgent:
    # Замена для high-level контроллера

    def __init__(self, agent, planner, replan_interval=200):
        self.agent = agent
        self.planner = planner
        self.config = agent.config
        self._step_count = 0
        self._replan_interval = replan_interval

    def infer_latent(self, batch):
        return self.agent.infer_latent(batch)

    def reset_episode(self):
        self.planner.reset()
        self._step_count = 0

    def sample_actions(self, observations, latents=None, seed=None,
                       temperature=1.0):
        obs = np.asarray(observations)
        if obs.ndim > 1:
            obs = obs[0]

        z_goal = jnp.array(latents)
        if z_goal.ndim > 1:
            z_goal = z_goal[0]

        # Перестраиваем план: в начале эпизода или каждые replan_interval шагов
        need_plan = (not self.planner.plan_active) or \
                    (self._step_count > 0 and
                     self._step_count % self._replan_interval == 0)

        if need_plan:
            self.planner.reset()
            self.planner.plan(obs, z_goal)

        z_subgoal = self.planner.get_current_intention(obs)
        self._step_count += 1

        low_dist = self.agent.network.select('actor')(
            jnp.array(obs), z_subgoal, goal_encoded=True,
            temperature=temperature
        )
        actions = low_dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)