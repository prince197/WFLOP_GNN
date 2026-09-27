import numpy as np


class GA:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        pc=0.8,
        pm=0.1,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm

        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        pop = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(ind)
            for ind in pop
        ])

        best_idx = np.argmin(fitness)

        best_pos = pop[best_idx].copy()
        best_score = fitness[best_idx]

        curve = [best_score]

        for _ in range(self.max_iter):

            new_pop = []

            while len(new_pop) < self.pop_size:

                # Tournament Selection
                i1, i2 = np.random.choice(
                    self.pop_size,
                    2,
                    replace=False
                )

                p1 = (
                    pop[i1]
                    if fitness[i1] < fitness[i2]
                    else pop[i2]
                )

                i1, i2 = np.random.choice(
                    self.pop_size,
                    2,
                    replace=False
                )

                p2 = (
                    pop[i1]
                    if fitness[i1] < fitness[i2]
                    else pop[i2]
                )

                c1 = p1.copy()
                c2 = p2.copy()

                # One-point crossover
                if np.random.rand() < self.pc:

                    point = np.random.randint(
                        1,
                        dim
                    )

                    c1[:point] = p1[:point]
                    c1[point:] = p2[point:]

                    c2[:point] = p2[:point]
                    c2[point:] = p1[point:]

                # Mutation
                for child in [c1, c2]:

                    mask = (
                        np.random.rand(dim)
                        < self.pm
                    )

                    child[mask] = np.random.uniform(
                        lb,
                        ub,
                        np.sum(mask)
                    )

                    child = np.clip(
                        child,
                        lb,
                        ub
                    )

                    new_pop.append(child)

                    if len(new_pop) >= self.pop_size:
                        break

            pop = np.array(new_pop)

            fitness = np.array([
                obj_fun(ind)
                for ind in pop
            ])

            idx = np.argmin(fitness)

            if fitness[idx] < best_score:

                best_score = fitness[idx]
                best_pos = pop[idx].copy()

            curve.append(best_score)

        return (
            best_pos,
            best_score,
            np.array(curve)
        )


class PSO:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        w=0.7,
        c1=2.0,
        c2=2.0,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter

        self.w = w
        self.c1 = c1
        self.c2 = c2

        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        X = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        V = np.zeros(
            (self.pop_size, dim)
        )

        pbest = X.copy()

        pbest_score = np.array([
            obj_fun(x)
            for x in X
        ])

        gbest_idx = np.argmin(
            pbest_score
        )

        gbest = pbest[gbest_idx].copy()

        gbest_score = pbest_score[
            gbest_idx
        ]

        curve = [gbest_score]

        for _ in range(self.max_iter):

            for i in range(
                self.pop_size
            ):

                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                V[i] = (
                    self.w * V[i]
                    + self.c1 * r1 * (pbest[i] - X[i])
                    + self.c2 * r2 * (gbest - X[i])
                )

                X[i] = X[i] + V[i]

                X[i] = np.clip(
                    X[i],
                    lb,
                    ub
                )

                score = obj_fun(X[i])

                if score < pbest_score[i]:

                    pbest_score[i] = score
                    pbest[i] = X[i].copy()

                if score < gbest_score:

                    gbest_score = score
                    gbest = X[i].copy()

            curve.append(gbest_score)

        return (
            gbest,
            gbest_score,
            np.array(curve)
        )
  

class DE:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        F=0.5,
        CR=0.9,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR

        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        pop = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(ind)
            for ind in pop
        ])

        best_idx = np.argmin(fitness)

        best_pos = pop[best_idx].copy()
        best_score = fitness[best_idx]

        curve = [best_score]

        for _ in range(self.max_iter):

            for i in range(self.pop_size):

                idxs = list(range(self.pop_size))
                idxs.remove(i)

                r1, r2, r3 = np.random.choice(
                    idxs,
                    3,
                    replace=False
                )

                mutant = (
                    pop[r1]
                    + self.F * (
                        pop[r2]
                        - pop[r3]
                    )
                )

                mutant = np.clip(
                    mutant,
                    lb,
                    ub
                )

                trial = pop[i].copy()

                j_rand = np.random.randint(
                    dim
                )

                for j in range(dim):

                    if (
                        np.random.rand()
                        < self.CR
                    ) or (
                        j == j_rand
                    ):

                        trial[j] = mutant[j]

                trial = np.clip(
                    trial,
                    lb,
                    ub
                )

                trial_fit = obj_fun(
                    trial
                )

                if trial_fit < fitness[i]:

                    pop[i] = trial
                    fitness[i] = trial_fit

                    if trial_fit < best_score:

                        best_score = trial_fit
                        best_pos = trial.copy()

            curve.append(
                best_score
            )

        return (
            best_pos,
            best_score,
            np.array(curve)
        )

class GWO:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter

        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        X = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(x)
            for x in X
        ])

        curve = []

        for t in range(
            self.max_iter
        ):

            idx = np.argsort(fitness)

            alpha = X[idx[0]]
            beta = X[idx[1]]
            delta = X[idx[2]]

            alpha_score = fitness[idx[0]]

            a = 2 - 2*t/self.max_iter

            for i in range(
                self.pop_size
            ):

                for j in range(dim):

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A1 = 2*a*r1 - a
                    C1 = 2*r2

                    D_alpha = abs(
                        C1*alpha[j]
                        - X[i,j]
                    )

                    X1 = (
                        alpha[j]
                        - A1*D_alpha
                    )

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A2 = 2*a*r1 - a
                    C2 = 2*r2

                    D_beta = abs(
                        C2*beta[j]
                        - X[i,j]
                    )

                    X2 = (
                        beta[j]
                        - A2*D_beta
                    )

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A3 = 2*a*r1 - a
                    C3 = 2*r2

                    D_delta = abs(
                        C3*delta[j]
                        - X[i,j]
                    )

                    X3 = (
                        delta[j]
                        - A3*D_delta
                    )

                    X[i,j] = (
                        X1 + X2 + X3
                    ) / 3

            X = np.clip(
                X,
                lb,
                ub
            )

            fitness = np.array([
                obj_fun(x)
                for x in X
            ])

            curve.append(
                alpha_score
            )

        best_idx = np.argmin(
            fitness
        )

        return (
            X[best_idx].copy(),
            fitness[best_idx],
            np.array(curve)
        )

class BBO:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        p_mutate=0.01,
        keep=2,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.p_mutate = p_mutate
        self.keep = keep

        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        pop = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        cost = np.array([
            obj_fun(ind)
            for ind in pop
        ])

        idx = np.argsort(cost)

        pop = pop[idx]
        cost = cost[idx]

        curve = [cost[0]]

        mu = (
            self.pop_size + 1
            - np.arange(1, self.pop_size + 1)
        ) / (self.pop_size + 1)

        lam = 1 - mu

        for it in range(self.max_iter):

            elite_pop = pop[:self.keep].copy()
            elite_cost = cost[:self.keep].copy()

            island = pop.copy()

            # Migration
            for k in range(self.pop_size):

                for j in range(dim):

                    if np.random.rand() < lam[k]:

                        probs = mu / np.sum(mu)

                        donor = np.random.choice(
                            self.pop_size,
                            p=probs
                        )

                        island[k, j] = pop[donor, j]

            # Mutation
            mutation_mask = (
                np.random.rand(
                    self.pop_size,
                    dim
                ) < self.p_mutate
            )

            island[mutation_mask] = np.random.uniform(
                lb,
                ub,
                np.sum(mutation_mask)
            )

            pop = island

            cost = np.array([
                obj_fun(ind)
                for ind in pop
            ])

            idx = np.argsort(cost)

            pop = pop[idx]
            cost = cost[idx]

            # Elitism
            pop[-self.keep:] = elite_pop
            cost[-self.keep:] = elite_cost

            idx = np.argsort(cost)

            pop = pop[idx]
            cost = cost[idx]

            curve.append(cost[0])

        return (
            pop[0].copy(),
            cost[0],
            np.array(curve)
        )

class SSA:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter

        if seed is not None:
            np.random.seed(seed)

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        pop = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(ind)
            for ind in pop
        ])

        idx = np.argsort(fitness)

        pop = pop[idx]
        fitness = fitness[idx]

        food_position = pop[0].copy()
        food_fitness = fitness[0]

        curve = [food_fitness]

        for t in range(1, self.max_iter + 1):

            c1 = 2 * np.exp(
                -(4 * t / self.max_iter) ** 2
            )

            new_pop = pop.copy()

            # -----------------------------
            # Leaders
            # -----------------------------
            for i in range(self.pop_size // 2):

                for j in range(dim):

                    c2 = np.random.rand()
                    c3 = np.random.rand()

                    step = c1 * (
                        (ub - lb) * c2 + lb
                    )

                    if c3 < 0.5:

                        new_pop[i, j] = (
                            food_position[j]
                            + step
                        )

                    else:

                        new_pop[i, j] = (
                            food_position[j]
                            - step
                        )

            # -----------------------------
            # Followers
            # -----------------------------
            for i in range(
                self.pop_size // 2,
                self.pop_size
            ):

                new_pop[i] = (
                    new_pop[i - 1]
                    + pop[i]
                ) / 2

            new_pop = np.clip(
                new_pop,
                lb,
                ub
            )

            pop = new_pop

            fitness = np.array([
                obj_fun(ind)
                for ind in pop
            ])

            idx = np.argsort(fitness)

            pop = pop[idx]
            fitness = fitness[idx]

            if fitness[0] < food_fitness:

                food_fitness = fitness[0]
                food_position = pop[0].copy()

            curve.append(food_fitness)

        return (
            food_position,
            food_fitness,
            np.array(curve)
        )

class LXSSA:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        phi=0.0,
        chi=1.0,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.phi = phi
        self.chi = chi

        if seed is not None:
            np.random.seed(seed)

    def laplace_random(self):

        z = np.random.rand()

        if z <= 0.5:
            gamma = self.phi + self.chi * np.log(2 * z)
        else:
            gamma = self.phi - self.chi * np.log(2 * (1 - z))

        return gamma

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        pop = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(ind)
            for ind in pop
        ])

        idx = np.argsort(fitness)

        pop = pop[idx]
        fitness = fitness[idx]

        food_position = pop[0].copy()
        food_fitness = fitness[0]

        curve = [food_fitness]

        for l in range(1, self.max_iter + 1):

            r1 = 2 * np.exp(
                -(4 * l / self.max_iter) ** 2
            )

            new_pop = pop.copy()

            half = self.pop_size // 2

            # ==========================
            # Leaders
            # ==========================
            for i in range(half):

                for j in range(dim):

                    r2 = np.random.rand()
                    r3 = np.random.rand()

                    step = r1 * (
                        (ub - lb) * r2 + lb
                    )

                    if r3 >= 0.5:

                        new_pop[i, j] = (
                            food_position[j]
                            + step
                        )

                    else:

                        new_pop[i, j] = (
                            food_position[j]
                            - step
                        )

            # ==========================
            # Followers
            # ==========================
            for i in range(half, self.pop_size):

                follower = (
                    pop[i]
                    + new_pop[i - 1]
                ) / 2.0

                gamma = self.laplace_random()

                current_salp = pop[i]

                candidate = (
                    current_salp
                    + gamma * (
                        food_position
                        - current_salp
                    )
                )

                follower = np.clip(
                    follower,
                    lb,
                    ub
                )

                candidate = np.clip(
                    candidate,
                    lb,
                    ub
                )

                if obj_fun(candidate) < obj_fun(follower):

                    new_pop[i] = candidate

                else:

                    new_pop[i] = follower

            new_pop = np.clip(
                new_pop,
                lb,
                ub
            )

            pop = new_pop

            fitness = np.array([
                obj_fun(ind)
                for ind in pop
            ])

            idx = np.argsort(fitness)

            pop = pop[idx]
            fitness = fitness[idx]

            if fitness[0] < food_fitness:

                food_fitness = fitness[0]
                food_position = pop[0].copy()

            curve.append(
                food_fitness
            )

        return (
            food_position,
            food_fitness,
            np.array(curve)
        )

class QASSA:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        seed=None
    ):

        self.pop_size = pop_size
        self.max_iter = max_iter

        if seed is not None:
            np.random.seed(seed)

    def quadratic_candidate(
        self,
        food_position,
        food_fitness,
        pop,
        fitness,
        dim
    ):

        idx = np.random.choice(
            np.arange(1, len(pop)),
            size=2,
            replace=False
        )

        B = pop[idx[0]]
        C = pop[idx[1]]

        fB = fitness[idx[0]]
        fC = fitness[idx[1]]
        fH = food_fitness

        z = np.zeros(dim)

        for j in range(dim):

            Hj = food_position[j]
            Bj = B[j]
            Cj = C[j]

            numerator = (
                (Bj**2 - Cj**2) * fH
                + (Cj**2 - Hj**2) * fB
                + (Hj**2 - Bj**2) * fC
            )

            denominator = (
                (Bj - Cj) * fH
                + (Cj - Hj) * fB
                + (Hj - Bj) * fC
            )

            if abs(denominator) < 1e-12:

                z[j] = Hj

            else:

                z[j] = 0.5 * (
                    numerator / denominator
                )

        return z

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        pop = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(ind)
            for ind in pop
        ])

        idx = np.argsort(fitness)

        pop = pop[idx]
        fitness = fitness[idx]

        food_position = pop[0].copy()
        food_fitness = fitness[0]

        curve = [food_fitness]

        for l in range(
            1,
            self.max_iter + 1
        ):

            g1 = 2 * np.exp(
                -(4 * l / self.max_iter) ** 2
            )

            new_pop = pop.copy()

            # Leaders
            for i in range(
                self.pop_size // 2
            ):

                for j in range(dim):

                    g2 = np.random.rand()
                    g3 = np.random.rand()

                    step = g1 * (
                        (ub - lb) * g2 + lb
                    )

                    if g3 >= 0.5:

                        new_pop[i, j] = (
                            food_position[j]
                            + step
                        )

                    else:

                        new_pop[i, j] = (
                            food_position[j]
                            - step
                        )

            # Followers
            for i in range(
                self.pop_size // 2,
                self.pop_size
            ):

                follower = (
                    pop[i]
                    + new_pop[i - 1]
                ) / 2.0

                qa_candidate = (
                    self.quadratic_candidate(
                        food_position,
                        food_fitness,
                        pop,
                        fitness,
                        dim
                    )
                )

                follower = np.clip(
                    follower,
                    lb,
                    ub
                )

                qa_candidate = np.clip(
                    qa_candidate,
                    lb,
                    ub
                )

                # Greedy Selection
                if obj_fun(
                    qa_candidate
                ) < obj_fun(
                    follower
                ):

                    new_pop[i] = (
                        qa_candidate
                    )

                else:

                    new_pop[i] = (
                        follower
                    )

            new_pop = np.clip(
                new_pop,
                lb,
                ub
            )

            pop = new_pop

            fitness = np.array([
                obj_fun(ind)
                for ind in pop
            ])

            idx = np.argsort(
                fitness
            )

            pop = pop[idx]
            fitness = fitness[idx]

            if (
                fitness[0]
                < food_fitness
            ):

                food_fitness = (
                    fitness[0]
                )

                food_position = (
                    pop[0].copy()
                )

            curve.append(
                food_fitness
            )

        return (
            food_position,
            food_fitness,
            np.array(curve)
        )

# ============================================================
# ACO  --  Ant Colony Optimization for WFLOP
# ------------------------------------------------------------
# Continuous adaptation of:
#   Eroglu, Y. & Seckiner, S.U. (2012). "Design of wind farm layout
#   using ant colony algorithm." Renewable Energy, 44, 53-62.
#
# This is NOT a textbook Ant System / ACS / MAX-MIN ant colony.
# Eroglu & Seckiner use a problem-specific "novel pheromone" scheme:
#
#   * The pheromone of a turbine is its own CONTRIBUTION to the total
#     wake loss of the farm. Badly-placed (high-wake-loss) turbines
#     therefore carry MORE pheromone.
#   * More ants are assigned to higher-pheromone turbines.
#   * Each assigned ant RELOCATES its turbine to a random feasible
#     position. The move is kept only if the farm objective improves
#     (greedy accept-if-improved).
#
# Adaptation to this codebase (as requested): positions stay CONTINUOUS
# inside the circular farm rather than snapping to a grid. The per-turbine
# wake-loss contribution is obtained as a leave-one-out marginal of the
# objective, so this class needs nothing but `obj_fun` and stays fully
# consistent with the identical objective used by every other optimizer
# in this file (same Jensen wake model, same penalties, minimization).
#
# Minimization; same interface and return signature as the other
# optimizers: optimize(obj_fun, dim, lb, ub) -> (best_pos, best_score, curve)
# ============================================================

class ACO:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        rho=0.1,
        p_global=0.5,
        sigma0=0.25,
        sigma_min=0.02,
        seed=None
    ):

        self.pop_size = pop_size      # number of ants dispatched per iteration
        self.max_iter = max_iter
        self.rho = rho                # pheromone evaporation / smoothing factor
        self.p_global = p_global      # prob. of a global (uniform-in-disk) relocation
        self.sigma0 = sigma0          # initial local relocation radius (fraction of farm radius)
        self.sigma_min = sigma_min    # final local relocation radius

        if seed is not None:
            np.random.seed(seed)

    # ---- uniform sample inside the circular farm ----
    def _sample_in_disk(self, radius):

        r = radius * np.sqrt(np.random.rand())
        phi = 2.0 * np.pi * np.random.rand()

        return np.array([
            r * np.cos(phi),
            r * np.sin(phi)
        ])

    # ---- pheromone = per-turbine contribution to total wake loss ----
    def _pheromone(self, x, N, obj_fun, f_full):
        """
        Leave-one-out marginal: how much of the farm's wake loss (plus the
        penalties it incurs) disappears if turbine i is removed.
        Larger value  ->  worse-placed turbine  ->  more pheromone  ->  more ants.
        """

        if N <= 1:
            return np.ones(N) / N

        coords = x.reshape(N, 2)

        tau = np.empty(N)

        for i in range(N):

            reduced = np.delete(coords, i, axis=0).ravel()

            tau[i] = f_full - obj_fun(reduced)

        # contributions must be non-negative to act as pheromone
        tau = np.maximum(tau, 0.0)

        total = np.sum(tau)

        if not np.isfinite(total) or total <= 1e-12:
            return np.ones(N) / N

        return tau / total

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        N = dim // 2

        farm_radius = ub          # run_experiments passes ub = farm_radius

        # -----------------------------
        # Initialise one layout inside the farm
        # -----------------------------
        current = np.concatenate([
            self._sample_in_disk(farm_radius)
            for _ in range(N)
        ])

        current_score = obj_fun(current)

        best_pos = current.copy()
        best_score = current_score

        curve = [best_score]

        # uniform prior on the pheromone before anything is known
        tau = np.ones(N) / N

        for t in range(1, self.max_iter + 1):

            # -----------------------------
            # Pheromone update: recompute per-turbine wake-loss shares,
            # then evaporate/smooth towards them.
            # -----------------------------
            tau_new = self._pheromone(current, N, obj_fun, current_score)

            tau = (1.0 - self.rho) * tau + self.rho * tau_new

            s = np.sum(tau)
            tau = tau / s if s > 1e-12 else np.ones(N) / N

            # -----------------------------
            # Assign more ants to higher-pheromone (worse) turbines
            # -----------------------------
            ant_counts = np.random.multinomial(self.pop_size, tau)

            # local relocation radius shrinks over the run
            frac = t / self.max_iter
            sigma_t = (
                self.sigma0
                + (self.sigma_min - self.sigma0) * frac
            ) * farm_radius

            # -----------------------------
            # Each ant relocates its turbine; keep the move only if the
            # farm objective improves (greedy accept-if-improved).
            # -----------------------------
            for i in range(N):

                for _ in range(ant_counts[i]):

                    candidate = current.copy()

                    if np.random.rand() < self.p_global:
                        # global move: anywhere in the farm
                        new_xy = self._sample_in_disk(farm_radius)
                    else:
                        # local move: Gaussian around the current position
                        new_xy = (
                            current[2*i : 2*i+2]
                            + np.random.randn(2) * sigma_t
                        )

                    candidate[2*i : 2*i+2] = np.clip(new_xy, lb, ub)

                    cand_score = obj_fun(candidate)

                    if cand_score < current_score:
                        current = candidate
                        current_score = cand_score

            if current_score < best_score:
                best_score = current_score
                best_pos = current.copy()

            curve.append(best_score)

        return (
            best_pos,
            best_score,
            np.array(curve)
        )


# ============================================================
# PF  --  Particle Filtering approach for WFLOP
# ------------------------------------------------------------
# Continuous adaptation of:
#   Eroglu, Y. & Seckiner, S.U. (2013). "Wind farm layout optimization
#   using particle filtering approach." Renewable Energy, 58, 95-107.
#
# That paper casts layout optimization as a state-estimation / filtering
# problem, following the particle-filter-for-optimization framework of
#   Zhou, E., Fu, M.C. & Marcus, S.I. (2008). "A particle filtering
#   framework for randomized optimization algorithms." Proc. Winter
#   Simulation Conference, 647-654.
#
# Each PARTICLE is a complete candidate layout. Each iteration:
#   1. PREDICT   : propagate particles by Gaussian system noise whose
#                  bandwidth dampens over the run (so the cloud settles).
#   2. WEIGHT    : w_i  proportional to  phi(H(x_i) - y_k), where H is the
#                  objective to maximize, y_k is a MONOTONICALLY INCREASING
#                  elite-quantile "observation", and phi is positive and
#                  strictly increasing. Particles below the elite quantile
#                  receive zero weight.
#   3. RESAMPLE  : systematic resampling on the normalized weights.
#
# Adaptation to this codebase (as requested): particles are CONTINUOUS
# coordinate vectors rather than grid indices. Because this project's
# objective is a MINIMIZATION of wake loss (and can return enormous
# penalty magnitudes), we set H = -f and normalize the weight exponent by
# the elite spread, which keeps the filter scale-invariant and stops the
# 1e10 penalties from collapsing every weight to zero.
#
# Minimization; same interface and return signature as the other
# optimizers: optimize(obj_fun, dim, lb, ub) -> (best_pos, best_score, curve)
# ============================================================

class PF:

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        elite_frac=0.20,
        sigma0=0.20,
        sigma_min=0.005,
        beta=5.0,
        seed=None
    ):

        self.pop_size = pop_size      # number of particles
        self.max_iter = max_iter
        self.elite_frac = elite_frac  # top fraction defining the observation y_k
        self.sigma0 = sigma0          # initial proposal bandwidth (fraction of span)
        self.sigma_min = sigma_min    # final proposal bandwidth
        self.beta = beta              # sharpness of phi (selection pressure)

        if seed is not None:
            np.random.seed(seed)

    def _systematic_resample(self, weights):
        """Standard systematic resampling; returns the chosen particle indices."""

        n = len(weights)

        positions = (np.random.rand() + np.arange(n)) / n

        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0          # guard against floating-point round-off

        idx = np.zeros(n, dtype=int)

        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative[j]:
                idx[i] = j
                i += 1
            else:
                j += 1

        return idx

    def optimize(
        self,
        obj_fun,
        dim,
        lb,
        ub
    ):

        span = ub - lb

        # -----------------------------
        # Initialise the particle cloud
        # -----------------------------
        particles = np.random.uniform(
            lb,
            ub,
            (self.pop_size, dim)
        )

        fitness = np.array([
            obj_fun(p)
            for p in particles
        ])

        best_idx = np.argmin(fitness)

        best_pos = particles[best_idx].copy()
        best_score = fitness[best_idx]

        curve = [best_score]

        # y_k must be monotonically increasing (Zhou et al., Assumption 2)
        y_prev = -np.inf

        n_elite = max(1, int(np.ceil(self.elite_frac * self.pop_size)))

        for t in range(1, self.max_iter + 1):

            # -----------------------------
            # (1) PREDICT: Gaussian propagation, dampening bandwidth
            # -----------------------------
            frac = t / self.max_iter
            sigma_t = (
                self.sigma0
                + (self.sigma_min - self.sigma0) * frac
            ) * span

            particles = np.clip(
                particles + np.random.randn(self.pop_size, dim) * sigma_t,
                lb,
                ub
            )

            fitness = np.array([
                obj_fun(p)
                for p in particles
            ])

            # -----------------------------
            # (2) WEIGHT: H = -f  (this project minimizes), elite quantile y_k
            # -----------------------------
            H = -fitness

            # track the incumbent BEFORE resampling reorders the cloud
            gen_best = np.argmin(fitness)

            if fitness[gen_best] < best_score:
                best_score = fitness[gen_best]
                best_pos = particles[gen_best].copy()

            # y_k = value of the n_elite-th best particle, forced non-decreasing
            y_k = np.sort(H)[-n_elite]

            if y_k < y_prev:
                y_k = y_prev

            y_prev = y_k

            elite = H >= y_k

            if not np.any(elite):
                weights = np.full(self.pop_size, 1.0 / self.pop_size)
            else:
                scale = H[elite].max() - y_k

                if not np.isfinite(scale) or scale <= 1e-12:
                    weights = elite.astype(float)
                else:
                    weights = np.zeros(self.pop_size)
                    weights[elite] = np.exp(
                        self.beta * (H[elite] - y_k) / scale
                    )

                w_sum = weights.sum()

                if not np.isfinite(w_sum) or w_sum <= 1e-12:
                    weights = elite.astype(float)
                    w_sum = weights.sum()

                weights = weights / w_sum

            # -----------------------------
            # (3) RESAMPLE: systematic resampling
            # -----------------------------
            idx = self._systematic_resample(weights)

            particles = particles[idx].copy()

            curve.append(best_score)

        return (
            best_pos,
            best_score,
            np.array(curve)
        )


# ============================================================
# GNNLXSSA -- GNN-enhanced Laplacian Salp Swarm Algorithm
# ------------------------------------------------------------
# PAPER-EXACT implementation of:
#   Solanki, P., Dwivedi, P., Garg, V., Shukla, V.
#   "Maximizing Wind Energy Efficiency with a Graph Neural Network
#    Enhanced Laplacian Salp Swarm Framework for Wind Farm Layout
#    Optimization."
#
# Faithful to the paper in every specified detail:
#   * GNWM surrogate (Sec. III-D2, Table III): custom message passing
#     (Eqs. 26-27) with weight-shared MLP message/update functions,
#     THREE layers, HIDDEN 64, ReLU, Adam lr 1e-3.
#   * Node features, Eq. (24) [7]: rho/r, sigma/r, (rho^2+sigma^2)/r^2,
#     zeta_bar, psi_bar, e_dir (2-D blowing-probability encoding).
#   * Edge features, Eq. (25) [5]: d_ij/r, beta_ij/alpha, drho/r,
#     dsigma/r, Jensen deficit delta_s_ij (physics-informed).
#     Directed edge (j -> i) iff beta_ij < alpha (Eq. 13), built once
#     with a direction-aggregated wake mask over the 24 bins.
#   * POWER HEAD, Eq. (28): per-node readout summed over nodes
#     (permutation-invariant; transfers across turbine counts N).
#     Here it predicts wake loss as % of ideal farm power - an affine
#     transform of the paper's normalized power P/P_ideal, so the
#     normalized-loss supervision of Eq. (30) is preserved.
#   * DIRECTION HEAD (Sec. III-D2b): per-node 2-D guidance vector
#     g_hat trained with the COSINE loss of Eq. (31) against
#     finite-difference improvement directions g_FD of the true
#     objective (computed on a subset of the pre-training buffer).
#     Total pre-training loss  L = L_power + lambda_g * L_dir
#     (Eq. 29), optimized jointly by manual backprop + Adam.
#   * GNN-LX-SSA loop (Algorithm 2): screening of ALL candidates by
#     the power head; exact evaluation of only the top-mu fraction
#     plus a random exploration sample; guided update, Eq. (32):
#     y = x + gamma(z, phi, chi) (H - x) + tau * g_hat, tau annealed
#     as tau0 (1 - t/T); leader update, Eq. (6), UNCHANGED; replay
#     buffer + periodic online fine-tuning of the power head.
#   * Constraint handling (Sec. III-E): boundary projection repair +
#     spacing displacement repair; the exact objective retains its
#     penalty terms (Deb-style dominance emerges from comparisons).
#
# Minimization; same interface and return signature as the other
# optimizers: optimize(obj_fun, dim, lb, ub) -> (best_pos, best_score,
# curve). best_score and curve are ALWAYS exact objective values.
# Bookkeeping after optimize(): n_exact_evals, n_surrogate_evals,
# n_finetunes.
# ============================================================

try:
    from objective import (
        R as _WTR, K as _WTK, CT as _WTCT,
        OMEGA_SCEN1, IDEAL_POWER_SCEN1, PSI_SCEN1,
        OMEGA_SCEN2, IDEAL_POWER_SCEN2, PSI_SCEN2,
        OMEGA_SCEN3, IDEAL_POWER_SCEN3, PSI_SCEN3
    )
except Exception:                                   # standalone fallback
    _WTR, _WTK, _WTCT = 38.5, 0.075, 0.8
    OMEGA_SCEN1 = np.array([0,0.01,0.01,0.01,0.01,0.20,0.60,0.01,0.01,0.01,
                            0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
                            0.01,0.01,0.01,0.01,0])
    IDEAL_POWER_SCEN1 = 14045.7374
    PSI_SCEN1 = np.full(24, 13.0)
    OMEGA_SCEN2, IDEAL_POWER_SCEN2, PSI_SCEN2 = OMEGA_SCEN1, IDEAL_POWER_SCEN1, PSI_SCEN1
    OMEGA_SCEN3, IDEAL_POWER_SCEN3, PSI_SCEN3 = OMEGA_SCEN1, IDEAL_POWER_SCEN1, PSI_SCEN1

_ALPHA_CONE = np.arctan(_WTK)
_AJ = 1.0 - np.sqrt(1.0 - _WTCT)
_THETAS = np.deg2rad(np.arange(0, 360, 15) + 7.5)   # 24 bin centres


# ---- Active wind data set (mutable module state; set per run) ----
# The GNWM graph features (direction weighting, wind-regime node feature)
# and the optimizer's power normalization depend on which wind data set is
# being solved. set_active_dataset() switches all of them consistently.
def _psi_bar(omega, psi):
    w = float(np.sum(omega))
    return float(np.sum(omega * psi) / w) if w > 0 else float(np.mean(psi))

_DATASETS = {
    1: dict(omega=OMEGA_SCEN1, ideal=IDEAL_POWER_SCEN1,
            psibar=_psi_bar(OMEGA_SCEN1, PSI_SCEN1)),
    2: dict(omega=OMEGA_SCEN2, ideal=IDEAL_POWER_SCEN2,
            psibar=_psi_bar(OMEGA_SCEN2, PSI_SCEN2)),
    3: dict(omega=OMEGA_SCEN3, ideal=IDEAL_POWER_SCEN3,
            psibar=_psi_bar(OMEGA_SCEN3, PSI_SCEN3)),   # Horns Rev 1
}

_OMEGA = _DATASETS[1]["omega"]      # active blowing-probability profile
_IDEALP = _DATASETS[1]["ideal"]     # active ideal power per turbine
_PSIBAR = _DATASETS[1]["psibar"]    # active direction-averaged scale


def set_active_dataset(dataset):
    """Select Wind Data Set 1, 2 or 3 (Horns Rev 1) for the surrogate/optimizer globals."""
    global _OMEGA, _IDEALP, _PSIBAR
    cfg = _DATASETS.get(int(dataset), _DATASETS[1])
    _OMEGA = cfg["omega"]
    _IDEALP = cfg["ideal"]
    _PSIBAR = cfg["psibar"]


class _GNWMSurrogate:
    """GNWM (paper Sec. III-D2): numpy message-passing network with a
    power head and a direction head, manual backprop, Adam."""

    NODE_F = 7          # Eq. (24)
    EDGE_F = 5          # Eq. (25)

    def __init__(self, hidden=64, layers=3, seed=0):
        rng = np.random.default_rng(seed)
        F, self.F, self.L = hidden, hidden, layers

        def xav(a, b):
            return rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b))

        self.p = {"Win": xav(self.NODE_F, F), "bin": np.zeros(F),
                  "V1": xav(F, F), "d1": np.zeros(F),        # power head
                  "V2": xav(F, 1), "d2": np.zeros(1),
                  "G1": xav(F, F), "e1": np.zeros(F),        # direction head
                  "G2": xav(F, 2), "e2": np.zeros(2)}
        for k in range(layers):
            self.p[f"W1{k}"] = xav(2 * F + self.EDGE_F, F)   # phi_e, Eq. (26)
            self.p[f"b1{k}"] = np.zeros(F)
            self.p[f"U1{k}"] = xav(2 * F, F)                 # phi_n, Eq. (27)
            self.p[f"c1{k}"] = np.zeros(F)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------- graph construction (Eqs. 24-25) ----------
    @staticmethod
    def build_graph(pos, r):
        N = len(pos)
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]      # rho_i - rho_j
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        RK = _WTR / _WTK

        sumw = np.zeros((N, N)); dbar = np.zeros((N, N))
        bbar = np.zeros((N, N)); sbar = np.zeros((N, N))
        for l, th in enumerate(_THETAS):
            w = _OMEGA[l]
            if w <= 0:
                continue
            proj = dx * np.cos(th) + dy * np.sin(th)
            d = np.abs(proj)
            den = np.sqrt((dx + RK * np.cos(th)) ** 2
                          + (dy + RK * np.sin(th)) ** 2)
            den = np.maximum(den, 1e-12)
            beta = np.arccos(np.clip((proj + RK) / den, -1, 1))
            mask = (beta < _ALPHA_CONE)
            np.fill_diagonal(mask, False)
            defc = _AJ / (1.0 + (_WTK / _WTR) * d) ** 2
            sumw += w * mask
            dbar += w * mask * d
            bbar += w * mask * beta
            sbar += w * mask * defc

        ii, jj = np.nonzero(sumw > 0)                 # i in wake of j
        E = len(ii)
        if E:
            sw = sumw[ii, jj]
            Eattr = np.stack([                        # Eq. (25), 5 features
                dbar[ii, jj] / sw / r,
                bbar[ii, jj] / sw / _ALPHA_CONE,
                (pos[ii, 0] - pos[jj, 0]) / r,
                (pos[ii, 1] - pos[jj, 1]) / r,
                sbar[ii, jj] / sw,
            ], axis=1)
        else:
            Eattr = np.zeros((0, _GNWMSurrogate.EDGE_F))

        edir = np.array([np.sum(_OMEGA * np.cos(_THETAS)),
                         np.sum(_OMEGA * np.sin(_THETAS))])
        Xn = np.column_stack([                        # Eq. (24), 7 features
            pos[:, 0] / r, pos[:, 1] / r,
            (pos[:, 0] ** 2 + pos[:, 1] ** 2) / r ** 2,
            np.full(N, 1.0),                          # zeta_bar / 2  (zeta=2)
            np.full(N, _PSIBAR / 13.0),               # psi_bar / 13  (dataset-dependent)
            np.full(N, edir[0]), np.full(N, edir[1])
        ])
        return Xn, jj, ii, Eattr                      # src=j (upstream), dst=i

    # ---------- forward: trunk + both heads ----------
    def forward(self, Xn, src, dst, Eattr):
        F, c = self.F, {}
        c["X"] = Xn
        z0 = Xn @ self.p["Win"] + self.p["bin"]
        h = np.maximum(z0, 0); c["z0"] = z0
        for k in range(self.L):
            if len(src):
                Min = np.concatenate([h[dst], h[src], Eattr], axis=1)
                zm = Min @ self.p[f"W1{k}"] + self.p[f"b1{k}"]
                m = np.maximum(zm, 0)
                agg = np.zeros_like(h)
                np.add.at(agg, dst, m)                # Eq. (26): sum over N(i)
            else:
                Min = np.zeros((0, 2 * F + self.EDGE_F))
                zm = np.zeros((0, F)); m = zm
                agg = np.zeros_like(h)
            Uin = np.concatenate([h, agg], axis=1)
            zu = Uin @ self.p[f"U1{k}"] + self.p[f"c1{k}"]   # Eq. (27)
            c[f"h{k}"], c[f"Min{k}"], c[f"zm{k}"] = h, Min, zm
            c[f"Uin{k}"], c[f"zu{k}"] = Uin, zu
            h = np.maximum(zu, 0)
        # power head, Eq. (28): per-node -> sum
        z1 = h @ self.p["V1"] + self.p["d1"]
        ph = np.maximum(z1, 0)
        pnode = ph @ self.p["V2"] + self.p["d2"]
        # direction head: per-node 2-D guidance
        zg = h @ self.p["G1"] + self.p["e1"]
        hg = np.maximum(zg, 0)
        gout = hg @ self.p["G2"] + self.p["e2"]
        c["hL"], c["z1"], c["ph"], c["pnode"] = h, z1, ph, pnode
        c["zg"], c["hg"], c["gout"] = zg, hg, gout
        c["src"], c["dst"] = src, dst
        return float(np.sum(pnode)), gout, c

    # ---------- backward through both heads + trunk ----------
    def backward(self, c, dP, dG=None):
        """dP: scalar dL/dP_hat. dG: (N,2) dL/dg_hat or None."""
        F, g = self.F, {}
        N = c["X"].shape[0]
        # power head
        dp = np.full((N, 1), dP)
        g["V2"] = c["ph"].T @ dp; g["d2"] = dp.sum(axis=0)
        dph = dp @ self.p["V2"].T
        dz1 = dph * (c["z1"] > 0)
        g["V1"] = c["hL"].T @ dz1; g["d1"] = dz1.sum(axis=0)
        dh = dz1 @ self.p["V1"].T
        # direction head
        if dG is not None:
            g["G2"] = c["hg"].T @ dG; g["e2"] = dG.sum(axis=0)
            dhg = dG @ self.p["G2"].T
            dzg = dhg * (c["zg"] > 0)
            g["G1"] = c["hL"].T @ dzg; g["e1"] = dzg.sum(axis=0)
            dh = dh + dzg @ self.p["G1"].T
        else:
            g["G1"] = np.zeros_like(self.p["G1"]); g["e1"] = np.zeros_like(self.p["e1"])
            g["G2"] = np.zeros_like(self.p["G2"]); g["e2"] = np.zeros_like(self.p["e2"])
        # trunk
        for k in reversed(range(self.L)):
            dzu = dh * (c[f"zu{k}"] > 0)
            g[f"U1{k}"] = c[f"Uin{k}"].T @ dzu
            g[f"c1{k}"] = dzu.sum(axis=0)
            dUin = dzu @ self.p[f"U1{k}"].T
            dh_prev = dUin[:, :F].copy()
            dagg = dUin[:, F:]
            src, dst = c["src"], c["dst"]
            if len(src):
                dm = dagg[dst]
                dzm = dm * (c[f"zm{k}"] > 0)
                g[f"W1{k}"] = c[f"Min{k}"].T @ dzm
                g[f"b1{k}"] = dzm.sum(axis=0)
                dMin = dzm @ self.p[f"W1{k}"].T
                np.add.at(dh_prev, dst, dMin[:, :F])
                np.add.at(dh_prev, src, dMin[:, F:2 * F])
            else:
                g[f"W1{k}"] = np.zeros_like(self.p[f"W1{k}"])
                g[f"b1{k}"] = np.zeros_like(self.p[f"b1{k}"])
            dh = dh_prev
        dz0 = dh * (c["z0"] > 0)
        g["Win"] = c["X"].T @ dz0; g["bin"] = dz0.sum(axis=0)
        return g

    # ---------- public helpers ----------
    def predict(self, pos, r):
        """Predicted wake loss in % of ideal farm power."""
        Xn, src, dst, Ea = self.build_graph(pos, r)
        P, _, _ = self.forward(Xn, src, dst, Ea)
        return P

    def guidance(self, pos, r):
        """(prediction, unit guidance vectors g_hat [N,2])."""
        Xn, src, dst, Ea = self.build_graph(pos, r)
        P, gout, _ = self.forward(Xn, src, dst, Ea)
        nrm = np.linalg.norm(gout, axis=1, keepdims=True)
        return P, gout / np.maximum(nrm, 1e-12)

    def _adam(self, acc, lr):
        self.t += 1
        b1c = 1 - 0.9 ** self.t
        b2c = 1 - 0.999 ** self.t
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * acc[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * acc[k] ** 2
            self.p[k] -= lr * (self.m[k] / b1c) / (
                np.sqrt(self.v[k] / b2c) + 1e-8)

    def train(self, samples, epochs=1, batch=16, lr=1e-3,
              lambda_g=0.5, rng=None):
        """samples: (Xn, src, dst, Eattr, target%, g_fd or None).
        Joint loss L = L_power + lambda_g L_dir (Eqs. 29-31)."""
        if not samples:
            return
        rng = rng or np.random.default_rng(0)
        idx = np.arange(len(samples))
        for _ in range(epochs):
            rng.shuffle(idx)
            for b0 in range(0, len(idx), batch):
                bidx = idx[b0:b0 + batch]
                acc = {k: np.zeros_like(v) for k, v in self.p.items()}
                for i in bidx:
                    Xn, src, dst, Ea, y, gfd = samples[i]
                    P, gout, c = self.forward(Xn, src, dst, Ea)
                    dP = 2.0 * (P - y) / len(bidx)          # Eq. (30)
                    dG = None
                    if gfd is not None:                     # Eq. (31)
                        Nn = len(Xn)
                        vn = np.linalg.norm(gout, axis=1, keepdims=True)
                        vn = np.maximum(vn, 1e-9)
                        vhat = gout / vn
                        dot = np.sum(gfd * vhat, axis=1, keepdims=True)
                        # d(1 - g.v/|v|)/dv = (-g + (g.vhat) vhat)/|v|
                        dG = (lambda_g / (Nn * len(bidx))) * \
                             (-(gfd) + dot * vhat) / vn
                    gs = self.backward(c, dP, dG)
                    for k in acc:
                        acc[k] += gs[k]
                self._adam(acc, lr)

    # ---------- persistence (offline pre-training, paper Sec. IV) ----------
    def save(self, path):
        import os
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez(path, **{k: np.asarray(v) for k, v in self.p.items()})

    def load(self, path):
        data = np.load(path)
        for k in self.p:
            if k in data:
                self.p[k] = data[k]
        # reset Adam state so online fine-tuning starts cleanly
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        return self


class GNNLXSSA:
    """GNN-LX-SSA, paper Algorithm 2 (see block comment above)."""

    def __init__(
        self,
        pop_size=50,
        max_iter=100,
        phi=0.0,
        chi=1.0,
        mu=0.2,
        tau0=0.05,
        hidden=64,            # Table III
        mp_layers=3,          # Table III
        n_pretrain=40,
        finetune_every=5,
        lambda_g=0.5,         # Eq. (29) weight [not specified in paper]
        fd_fraction=0.25,     # share of pretrain samples given FD labels
        fd_max=16,            # cap on FD-labelled samples (4N evals each)
        backend="auto",       # "auto" | "pyg" | "numpy" | "mlp"
        pretrained_path=None, # offline-pre-trained GNWM (paper Sec. IV)
        dataset=1,            # 1 = Wind Data Set I, 2 = Wind Data Set II
        seed=None
    ):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.phi = phi                  # Laplace location, Eq. (9)
        self.chi = chi                  # Laplace scale, Eq. (9)
        self.mu = mu                    # exact-evaluation fraction
        self.tau0 = tau0                # guidance coefficient (annealed)
        self.hidden = hidden
        self.mp_layers = mp_layers
        self.n_pretrain = n_pretrain
        self.finetune_every = finetune_every
        self.lambda_g = lambda_g
        self.fd_fraction = fd_fraction
        self.fd_max = fd_max
        self.backend = backend
        self.pretrained_path = pretrained_path
        self.dataset = dataset
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    # ---------------- geometry / constraint helpers ----------------
    @staticmethod
    def _sample_in_disk(r):
        rad = r * np.sqrt(np.random.rand())
        ang = 2 * np.pi * np.random.rand()
        return np.array([rad * np.cos(ang), rad * np.sin(ang)])

    @staticmethod
    def _boundary_repair(pos, r):
        nrm = np.sqrt(np.sum(pos ** 2, axis=1))
        out = nrm > r
        if np.any(out):
            pos[out] *= (r / nrm[out])[:, None]
        return pos

    @staticmethod
    def _spacing_repair(pos, r, passes=2):
        mind = 8.0 * _WTR
        N = len(pos)
        for _ in range(passes):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    v = pos[i] - pos[j]
                    d = np.linalg.norm(v)
                    if d < mind:
                        moved = True
                        u = v / d if d > 1e-9 else np.array([1.0, 0.0])
                        shift = 0.5 * (mind - d) + 1e-6
                        pos[i] += u * shift
                        pos[j] -= u * shift
            if not moved:
                break
            GNNLXSSA._boundary_repair(pos, r)
        return pos

    def _feasible_layout(self, N, r, tries=200):
        dmin = 8.0 * _WTR
        # fast rejection sampling (works for small/medium N)
        pts = []
        for _ in range(max(tries, 30 * N)):
            cand = self._sample_in_disk(r)
            if all(np.linalg.norm(cand - q) >= dmin for q in pts):
                pts.append(cand)
                if len(pts) == N:
                    return np.array(pts)
        # constructive concentric-ring fallback (scales to N >= 100):
        # radial gap 1.02 dmin guarantees inter-ring spacing; angular
        # spacing 1.05 dmin guarantees same-ring chords >= dmin
        grid = [np.zeros(2)]
        k = 1
        while k * dmin * 1.02 <= r * 0.995:
            R_k = k * dmin * 1.02
            m = max(1, int(np.floor(2 * np.pi * R_k / (dmin * 1.05))))
            a0 = np.random.rand() * 2 * np.pi
            for a in np.linspace(0, 2 * np.pi, m, endpoint=False):
                grid.append(R_k * np.array([np.cos(a + a0),
                                            np.sin(a + a0)]))
            k += 1
        grid = np.array(grid)
        if len(grid) >= N:
            sel = np.random.choice(len(grid), size=N, replace=False)
            lay = grid[sel] + np.random.normal(0, dmin * 0.02, (N, 2))
            lay = self._boundary_repair(lay, r)
            return self._spacing_repair(lay, r, passes=8)
        # farm too small for N at this spacing: best effort
        while len(pts) < N:
            pts.append(self._sample_in_disk(r))
        return self._spacing_repair(np.array(pts), r, passes=8)

    # ---------------- finite-difference direction labels ----------------
    @staticmethod
    def _fd_direction(obj_fun, x, N, h=5.0):
        """g_FD (Sec. III-D3): unit IMPROVEMENT direction of the true
        objective (descent, since the framework minimizes wake loss)."""
        g = np.zeros((N, 2))
        for i in range(N):
            for cdx in range(2):
                p = x.copy(); p[2 * i + cdx] += h
                m = x.copy(); m[2 * i + cdx] -= h
                g[i, cdx] = (obj_fun(p) - obj_fun(m)) / (2 * h)
        g = -g                                        # descend the loss
        nrm = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(nrm, 1e-12)

    # ---------------- main loop (Algorithm 2) ----------------
    def optimize(self, obj_fun, dim, lb, ub):

        N = dim // 2
        r = ub
        set_active_dataset(self.dataset)     # switch omega / ideal / psi_bar
        gnn = make_gnwm(self.hidden, self.mp_layers,
                        seed=self.seed if self.seed is not None else 0,
                        backend=self.backend)
        rng = np.random.default_rng(self.seed if self.seed is not None else 0)

        # Load an offline-pre-trained GNWM if available (paper Sec. IV:
        # 50k layouts, 70/15/15 split, 100 epochs, batch 64). When present,
        # the lightweight inline pre-training below is skipped.
        pretrained = False
        if self.pretrained_path and hasattr(gnn, "load"):
            import os as _os
            if _os.path.exists(self.pretrained_path):
                gnn.load(self.pretrained_path)
                pretrained = True

        norm = 100.0 / (_IDEALP * N)     # exact -> % of ideal power
        feas_cap = _IDEALP * N           # penalty-contaminated filter

        buffer = []                      # (Xn, src, dst, Ea, target%, g_fd)
        self.n_exact_evals = 0
        self.n_surrogate_evals = 0
        self.n_finetunes = 0

        def exact(x):
            self.n_exact_evals += 1
            return obj_fun(x)

        def add_sample(pos, f, gfd=None):
            if f < feas_cap:
                Xn, s, d, Ea = gnn.build_graph(pos, r)
                buffer.append((Xn, s, d, Ea, f * norm, gfd))

        # ---- initial population (Algorithm 2, lines 1-2) ----
        pop = np.array([self._feasible_layout(N, r).ravel()
                        for _ in range(self.pop_size)])
        fit = np.empty(self.pop_size)
        for i in range(self.pop_size):
            fit[i] = exact(pop[i])
            add_sample(pop[i].reshape(N, 2), fit[i])

        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        best_pos, best_score = pop[0].copy(), fit[0]
        H = best_pos.copy()
        curve = [best_score]

        # ---- pre-training ----
        # Preferred path: an offline-pre-trained GNWM loaded above (paper
        # Sec. IV). Fallback (no pretrained file): a lightweight inline
        # pre-training on random feasible layouts, an fd_fraction subset
        # (capped at fd_max) receiving finite-difference labels.
        if not pretrained:
            n_fd = min(self.fd_max, max(1, int(self.fd_fraction *
                                               (self.n_pretrain + self.pop_size))))
            fd_assigned = 0
            for k in range(self.n_pretrain):
                lay = self._feasible_layout(N, r)
                x = lay.ravel()
                f = exact(x)
                gfd = None
                if fd_assigned < n_fd and f < feas_cap:
                    gfd = self._fd_direction(obj_fun, x, N)
                    self.n_exact_evals += 4 * N
                    fd_assigned += 1
                add_sample(lay, f, gfd)
                if f < best_score:
                    best_score = f
                    best_pos = x.copy()
                    H = best_pos.copy()
            gnn.train(buffer, epochs=25, batch=16,
                      lambda_g=self.lambda_g, rng=rng)

        n_exact = max(1, int(np.ceil(self.mu * 2 * self.pop_size)))

        for t in range(1, self.max_iter + 1):

            r1 = 2.0 * np.exp(-(4.0 * t / self.max_iter) ** 2)
            tau = self.tau0 * (1.0 - t / self.max_iter)     # annealed

            # ---- LX-SSA moves: leader Eq. (6) UNCHANGED, follower Eq. (7) ----
            moved = pop.copy()
            for i in range(self.pop_size):
                if i == 0:
                    r2 = np.random.rand(dim)
                    r3 = np.random.rand(dim)
                    step = r1 * ((ub - lb) * r2 + lb)
                    moved[i] = np.where(r3 >= 0.5, H + step, H - step)
                else:
                    moved[i] = 0.5 * (pop[i] + moved[i - 1])

            # ---- guided Laplace candidates, Eq. (32) ----
            # g_hat comes from the TRAINED DIRECTION HEAD (paper
            # Sec. III-D2b), not from a surrogate input gradient.
            cands = np.empty_like(moved)
            for i in range(self.pop_size):
                z = np.random.rand()
                if z <= 0.5:
                    gamma = self.phi - self.chi * np.log(max(z, 1e-12))
                else:
                    gamma = self.phi + self.chi * np.log(z)
                _, ghat = gnn.guidance(moved[i].reshape(N, 2), r)
                self.n_surrogate_evals += 1
                cands[i] = (moved[i]
                            + gamma * (H - moved[i])
                            + (tau * r) * ghat.ravel())

            # ---- repair both sets (Sec. III-E, unchanged) ----
            for arr in (moved, cands):
                for i in range(self.pop_size):
                    p2 = arr[i].reshape(N, 2)
                    self._boundary_repair(p2, r)
                    self._spacing_repair(p2, r, passes=1)
                    arr[i] = np.clip(p2.ravel(), lb, ub)

            # ---- surrogate screening of all 2*Np candidates ----
            allc = np.vstack([moved, cands])
            pred = np.array([gnn.predict(x.reshape(N, 2), r) for x in allc])
            self.n_surrogate_evals += len(allc)

            exact_idx = set(np.argsort(pred)[:n_exact].tolist())
            exact_idx |= set(rng.choice(len(allc), size=2,
                                        replace=False).tolist())

            score = pred / norm                     # surrogate, exact units
            for idx in exact_idx:
                f = exact(allc[idx])
                score[idx] = f
                add_sample(allc[idx].reshape(N, 2), f)
                if f < best_score:
                    best_score = f
                    best_pos = allc[idx].copy()

            # ---- greedy selection per salp (Algorithm 2, line 21) ----
            for i in range(self.pop_size):
                a, b = i, i + self.pop_size
                keep = a if score[a] <= score[b] else b
                pop[i] = allc[keep]
                fit[i] = score[keep]

            H = best_pos.copy()

            # ---- periodic online fine-tuning (Algorithm 2, line 24) ----
            if t % self.finetune_every == 0 and buffer:
                take = min(len(buffer), 96)
                sub = [buffer[k] for k in
                       rng.choice(len(buffer), size=take, replace=False)]
                gnn.train(sub, epochs=2, batch=16,
                          lambda_g=self.lambda_g, rng=rng)
                self.n_finetunes += 1

            curve.append(best_score)

        return (
            best_pos,
            best_score,
            np.array(curve)
        )


# ============================================================
# Surrogate backend factory + MLP ablation baseline
# ============================================================

def make_gnwm(hidden=64, layers=3, seed=0, backend="auto"):
    """
    backend:
      "auto"  - PyTorch Geometric if importable, else numpy
      "pyg"   - PyTorch Geometric (raises if unavailable)
      "numpy" - manual-backprop numpy GNWM (no dependencies)
      "mlp"   - coordinate MLP, NO graph structure (ablation:
                'why use a graph instead of an MLP?')
    All backends expose the same interface.
    """
    if backend == "mlp":
        return _MLPSurrogate(hidden=hidden, seed=seed)
    if backend in ("auto", "pyg"):
        try:
            from surrogate_pyg import GNWMSurrogatePyG, PYG_AVAILABLE
            if PYG_AVAILABLE:
                return GNWMSurrogatePyG(hidden=hidden, layers=layers,
                                        seed=seed)
            if backend == "pyg":
                raise ImportError("torch/torch-geometric not available")
        except ImportError:
            if backend == "pyg":
                raise
    return _GNWMSurrogate(hidden=hidden, layers=layers, seed=seed)


class _MLPSurrogate:
    """Graph-free ablation surrogate: 2-hidden-layer MLP on angle-sorted
    normalized coordinates. Same interface as _GNWMSurrogate; guidance
    vectors are zero (no per-node structure to read a direction from)."""

    def __init__(self, hidden=64, seed=0):
        self.F = hidden
        self.rng = np.random.default_rng(seed)
        self.p = None            # lazy init once N is known
        self.t = 0

    @staticmethod
    def build_graph(pos, r):
        order = np.argsort(np.arctan2(pos[:, 1], pos[:, 0]))
        feat = (pos[order] / r).ravel()
        return feat, np.zeros(0, int), np.zeros(0, int), np.zeros((0, 5))

    def _init(self, d_in):
        F = self.F
        def xav(a, b):
            return self.rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b))
        self.p = {"W1": xav(d_in, F), "b1": np.zeros(F),
                  "W2": xav(F, F), "b2": np.zeros(F),
                  "W3": xav(F, 1), "b3": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}

    def _forward(self, x):
        z1 = x @ self.p["W1"] + self.p["b1"]; h1 = np.maximum(z1, 0)
        z2 = h1 @ self.p["W2"] + self.p["b2"]; h2 = np.maximum(z2, 0)
        out = h2 @ self.p["W3"] + self.p["b3"]
        return float(out[0]), (x, z1, h1, z2, h2)

    def predict(self, pos, r):
        feat, *_ = self.build_graph(pos, r)
        if self.p is None:
            self._init(len(feat))
        return self._forward(feat)[0]

    def guidance(self, pos, r):
        return self.predict(pos, r), np.zeros((len(pos), 2))

    def train(self, samples, epochs=1, batch=16, lr=1e-3,
              lambda_g=0.5, rng=None):
        if not samples:
            return
        rng = rng or np.random.default_rng(0)
        if self.p is None:
            self._init(len(samples[0][0]))
        idx = np.arange(len(samples))
        for _ in range(epochs):
            rng.shuffle(idx)
            for b0 in range(0, len(idx), batch):
                bidx = idx[b0:b0 + batch]
                acc = {k: np.zeros_like(v) for k, v in self.p.items()}
                for i in bidx:
                    feat, _, _, _, y, _ = samples[i]
                    P, (x, z1, h1, z2, h2) = self._forward(feat)
                    d = 2.0 * (P - y) / len(bidx)
                    acc["W3"] += np.outer(h2, [d]); acc["b3"] += d
                    dh2 = d * self.p["W3"][:, 0]
                    dz2 = dh2 * (z2 > 0)
                    acc["W2"] += np.outer(h1, dz2); acc["b2"] += dz2
                    dh1 = dz2 @ self.p["W2"].T
                    dz1 = dh1 * (z1 > 0)
                    acc["W1"] += np.outer(x, dz1); acc["b1"] += dz1
                self.t += 1
                b1c = 1 - 0.9 ** self.t; b2c = 1 - 0.999 ** self.t
                for k in self.p:
                    self.m[k] = 0.9 * self.m[k] + 0.1 * acc[k]
                    self.v[k] = 0.999 * self.v[k] + 0.001 * acc[k] ** 2
                    self.p[k] -= lr * (self.m[k] / b1c) / (
                        np.sqrt(self.v[k] / b2c) + 1e-8)


# ============================================================
# GNNLXSSA_UQ -- deep-ensemble, UNCERTAINTY-GUIDED GNN-LX-SSA
# ------------------------------------------------------------
# Framework variant in which the trust decision is made by a deep
# ensemble instead of top-mu screening:
#   * n_models independently initialised GNWM surrogates (different
#     weight seeds + shuffling; identical training data).
#   * For each candidate: ensemble mean m and std s (both in % of
#     ideal farm power). If s < sigma_threshold the candidate is
#     accepted on the surrogate mean; otherwise it is evaluated
#     EXACTLY, the sample enters the replay buffer, and after every
#     `retrain_interval` new exact samples the whole ensemble is
#     fine-tuned.
#   * The generation-best candidate is ALWAYS verified exactly, so
#     the reported best/curve are ground truth.
#   * Guidance: ensemble-averaged direction-head output.
#   * self.uq_records collects (mean, std, exact) triples in EXACT
#     units (kW) whenever an exact evaluation follows a prediction -
#     the raw material for calibration / coverage / sigma-vs-error
#     validation (see uncertainty_validation.py).
# ============================================================

class GNNLXSSA_UQ(GNNLXSSA):

    def __init__(self, *args, n_models=5, sigma_threshold=0.5,
                 retrain_interval=20, retrain_epochs=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_models = n_models
        self.sigma_threshold = sigma_threshold    # in % of ideal power
        self.retrain_interval = retrain_interval
        self.retrain_epochs = retrain_epochs

    # ---------------- ensemble helpers ----------------
    def _ens_predict(self, ens, pos, r):
        preds = np.array([g.predict(pos, r) for g in ens])
        return float(preds.mean()), float(preds.std())

    def _ens_guidance(self, ens, pos, r):
        gs = []
        for g in ens:
            _, gh = g.guidance(pos, r)
            gs.append(gh)
        g = np.mean(gs, axis=0)
        nrm = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(nrm, 1e-12)

    # ---------------- main loop ----------------
    def optimize(self, obj_fun, dim, lb, ub):

        N = dim // 2
        r = ub
        set_active_dataset(self.dataset)     # switch omega / ideal / psi_bar
        base_seed = self.seed if self.seed is not None else 0
        ens = [make_gnwm(self.hidden, self.mp_layers,
                         seed=base_seed + 31 * m, backend=self.backend)
               for m in range(self.n_models)]
        rng = np.random.default_rng(base_seed)

        # Warm-start every ensemble member from the offline-pre-trained
        # GNWM if available (paper Sec. IV); ensemble diversity then comes
        # from independent online fine-tuning seeds.
        pretrained = False
        if self.pretrained_path:
            import os as _os
            if _os.path.exists(self.pretrained_path):
                loaded = [g.load(self.pretrained_path)
                          for g in ens if hasattr(g, "load")]
                pretrained = len(loaded) == len(ens)

        norm = 100.0 / (_IDEALP * N)
        feas_cap = _IDEALP * N

        buffer = []
        self.n_exact_evals = 0
        self.n_surrogate_evals = 0
        self.n_retrains = 0
        self.uq_records = []          # (mean_kW, std_kW, exact_kW)
        since_retrain = 0

        def exact(x):
            self.n_exact_evals += 1
            return obj_fun(x)

        def add_sample(pos, f, gfd=None):
            if f < feas_cap:
                Xn, s, d, Ea = ens[0].build_graph(pos, r)
                buffer.append((Xn, s, d, Ea, f * norm, gfd))

        # ---- initial population, all exact ----
        pop = np.array([self._feasible_layout(N, r).ravel()
                        for _ in range(self.pop_size)])
        fit = np.empty(self.pop_size)
        for i in range(self.pop_size):
            fit[i] = exact(pop[i])
            add_sample(pop[i].reshape(N, 2), fit[i])
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        best_pos, best_score = pop[0].copy(), fit[0]
        H = best_pos.copy()
        curve = [best_score]

        # ---- pre-training (fallback only; skipped if warm-started) ----
        if not pretrained:
            n_fd = min(self.fd_max, max(1, int(self.fd_fraction *
                                               (self.n_pretrain + self.pop_size))))
            fd_assigned = 0
            for k in range(self.n_pretrain):
                lay = self._feasible_layout(N, r)
                x = lay.ravel()
                f = exact(x)
                gfd = None
                if fd_assigned < n_fd and f < feas_cap:
                    gfd = self._fd_direction(obj_fun, x, N)
                    self.n_exact_evals += 4 * N
                    fd_assigned += 1
                add_sample(lay, f, gfd)
                if f < best_score:
                    best_score, best_pos = f, x.copy()
                    H = best_pos.copy()
            for m, g in enumerate(ens):
                g.train(buffer, epochs=25, batch=16, lambda_g=self.lambda_g,
                        rng=np.random.default_rng(base_seed + 977 * m))

        for t in range(1, self.max_iter + 1):

            r1 = 2.0 * np.exp(-(4.0 * t / self.max_iter) ** 2)
            tau = self.tau0 * (1.0 - t / self.max_iter)

            # ---- LX-SSA moves (identical to GNNLXSSA) ----
            moved = pop.copy()
            for i in range(self.pop_size):
                if i == 0:
                    r2 = np.random.rand(dim)
                    r3 = np.random.rand(dim)
                    step = r1 * ((ub - lb) * r2 + lb)
                    moved[i] = np.where(r3 >= 0.5, H + step, H - step)
                else:
                    moved[i] = 0.5 * (pop[i] + moved[i - 1])

            cands = np.empty_like(moved)
            for i in range(self.pop_size):
                z = np.random.rand()
                if z <= 0.5:
                    gamma = self.phi - self.chi * np.log(max(z, 1e-12))
                else:
                    gamma = self.phi + self.chi * np.log(z)
                ghat = self._ens_guidance(ens, moved[i].reshape(N, 2), r)
                self.n_surrogate_evals += self.n_models
                cands[i] = (moved[i]
                            + gamma * (H - moved[i])
                            + (tau * r) * ghat.ravel())

            for arr in (moved, cands):
                for i in range(self.pop_size):
                    p2 = arr[i].reshape(N, 2)
                    self._boundary_repair(p2, r)
                    self._spacing_repair(p2, r, passes=1)
                    arr[i] = np.clip(p2.ravel(), lb, ub)

            # ---- UNCERTAINTY GATE over all 2*Np candidates ----
            allc = np.vstack([moved, cands])
            score = np.empty(len(allc))
            for idx in range(len(allc)):
                pos2 = allc[idx].reshape(N, 2)
                mean_pct, std_pct = self._ens_predict(ens, pos2, r)
                self.n_surrogate_evals += self.n_models
                if std_pct >= self.sigma_threshold:
                    f = exact(allc[idx])
                    self.uq_records.append(
                        (mean_pct / norm, std_pct / norm, f))
                    score[idx] = f
                    add_sample(pos2, f)
                    since_retrain += 1
                    if f < best_score:
                        best_score, best_pos = f, allc[idx].copy()
                else:
                    score[idx] = mean_pct / norm

            # ---- verify the generation best exactly (trust anchor) ----
            gb = int(np.argmin(score))
            mean_kW = score[gb]
            f = exact(allc[gb])
            score[gb] = f
            add_sample(allc[gb].reshape(N, 2), f)
            if f < best_score:
                best_score, best_pos = f, allc[gb].copy()

            # ---- greedy selection per salp ----
            for i in range(self.pop_size):
                a, b = i, i + self.pop_size
                keep = a if score[a] <= score[b] else b
                pop[i] = allc[keep]
                fit[i] = score[keep]

            H = best_pos.copy()

            # ---- retrain the ensemble on fresh exact samples ----
            if since_retrain >= self.retrain_interval and buffer:
                take = min(len(buffer), 96)
                for m, g in enumerate(ens):
                    sub_rng = np.random.default_rng(
                        base_seed + 977 * m + t)
                    sub = [buffer[k] for k in
                           sub_rng.choice(len(buffer), size=take,
                                          replace=False)]
                    g.train(sub, epochs=self.retrain_epochs, batch=16,
                            lambda_g=self.lambda_g, rng=sub_rng)
                self.n_retrains += 1
                since_retrain = 0

            curve.append(best_score)

        return best_pos, best_score, np.array(curve)
