import ext_elev

id = ["000000000"]


class Controller:
    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.capacities = self.game.get_capacities()
        self.reachable = self.game.get_reachable()
        self.elev_ids = tuple(sorted(self.capacities))
        self.eidx = {e: i for i, e in enumerate(self.elev_ids)}
        self.goal_reward = float(self.game.get_goal_reward())

        initial_state = self.game.get_initial_state()
        self.initial_persons = tuple(pid for pid, _ in initial_state[1])
        self.person_weight = {p: self.game.get_person_weight(p) for p in self.initial_persons}
        self.person_goal = {p: self.game.get_person_goal(p) for p in self.initial_persons}

        self.move_stats = {e: [2.0, 1.0] for e in self.elev_ids}
        self.person_stats = {p: [2.0, 1.0] for p in self.initial_persons}
        self.reward_sum = {p: 0.0 for p in self.initial_persons}
        self.reward_n = {p: 0 for p in self.initial_persons}

        self.prev_state = None
        self.prev_action = None

        self.farm_person = None
        self._path_cache = {}

    def choose_next_action(self, state):
        self._learn_from_last_step(state)
        legal = self._legal_actions(state)

        if self.farm_person is None:
            self._maybe_enable_farming(state)

        chosen = None
        if self.farm_person is not None:
            chosen = self._choose_farm_action(state, legal)
        if chosen is None:
            chosen = self._choose_goal_action(state)
        if chosen not in legal:
            chosen = "RESET" if "RESET" in legal else legal[0]

        self.prev_state = state
        self.prev_action = chosen
        return chosen

    def _p_move(self, e):
        s, f = self.move_stats[e]
        return max(0.05, min(0.995, s / (s + f)))

    def _p_person(self, p):
        s, f = self.person_stats[p]
        return max(0.05, min(0.995, s / (s + f)))

    def _reward_est(self, p):
        if self.reward_n[p] == 0:
            return 6.0
        return self.reward_sum[p] / self.reward_n[p]

    def _choose_goal_action(self, state):
        persons_map = dict(state[1])
        best = None
        best_score = -1.0
        for p in persons_map:
            cost, action = self._best_first_action_for_person(state, p)
            if action is None or cost <= 0.0:
                continue
            reward = self._reward_est(p)
            if state[2] == 1:
                reward += self.goal_reward
            score = reward / cost
            if score > best_score:
                best_score = score
                best = action
        return best if best is not None else "RESET"

    def _choose_farm_action(self, state, legal):
        persons_map = dict(state[1])
        if self.farm_person not in persons_map:
            return "RESET"
        _, action = self._best_first_action_for_person(state, self.farm_person)
        if action in legal:
            return action
        return "RESET"

    def _maybe_enable_farming(self, state):
        if self.game.get_current_steps() < 20:
            return
        init = self.game.get_initial_state()
        init_persons = dict(init[1])
        if not init_persons:
            return
        best_p = None
        best_rate = -1.0
        for p in init_persons:
            cost, _ = self._best_first_action_for_person(init, p)
            if cost <= 0:
                continue
            loop_cost = cost + 1.0
            rate = self._reward_est(p) / loop_cost
            if rate > best_rate:
                best_rate = rate
                best_p = p
        full_reward = self.goal_reward + sum(self._reward_est(p) for p in init_persons)
        full_cost = 0.0
        for p in init_persons:
            c, _ = self._best_first_action_for_person(init, p)
            full_cost += max(c, 1.0)
        full_rate = full_reward / max(1.0, full_cost)
        if best_p is not None and best_rate > 1.35 * full_rate:
            self.farm_person = best_p

    def _best_first_action_for_person(self, state, p):
        elevators_t, persons_t, _ = state
        persons_map = dict(persons_t)
        if p not in persons_map:
            return 0.0, None
        p_loc = persons_map[p]
        floors = tuple(fl for _, fl, _ in sorted(elevators_t))
        loads = tuple(w for _, _, w in sorted(elevators_t))
        key = (p, p_loc, floors, loads)
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached

        goal = self.person_goal[p]
        pprob = self._p_person(p)
        enter_exit_cost = 1.0 / pprob
        base_load = {eid: w for eid, _, w in elevators_t}
        weight = self.person_weight[p]

        start_node = (floors, p_loc[0], p_loc[1])
        dist = {start_node: 0.0}
        first_action = {start_node: None}
        pq = [(0.0, start_node)]
        best_terminal_cost = None
        best_terminal_action = None

        import heapq
        while pq:
            cur_d, node = heapq.heappop(pq)
            if cur_d != dist.get(node):
                continue
            ftuple, ltype, lval = node
            if best_terminal_cost is not None and cur_d >= best_terminal_cost:
                continue

            for e in self.elev_ids:
                i = self.eidx[e]
                cur_floor = ftuple[i]
                pmove = self._p_move(e)
                move_cost = 1.0 / pmove
                for trg in self.reachable[e]:
                    if trg == cur_floor:
                        continue
                    nft = list(ftuple)
                    nft[i] = trg
                    nft = tuple(nft)
                    nxt = (nft, ltype, lval)
                    nd = cur_d + move_cost
                    a = f"MOVE{{{e},{trg}}}"
                    fa = a if first_action[node] is None else first_action[node]
                    if nd < dist.get(nxt, float("inf")):
                        dist[nxt] = nd
                        first_action[nxt] = fa
                        heapq.heappush(pq, (nd, nxt))

            if ltype == "floor":
                floor = lval
                for e in self.elev_ids:
                    if floor not in self.reachable[e]:
                        continue
                    i = self.eidx[e]
                    if ftuple[i] != floor:
                        continue
                    if base_load[e] + weight > self.capacities[e]:
                        continue
                    nxt = (ftuple, "in", e)
                    nd = cur_d + enter_exit_cost
                    a = f"ENTER{{{p},{e}}}"
                    fa = a if first_action[node] is None else first_action[node]
                    if nd < dist.get(nxt, float("inf")):
                        dist[nxt] = nd
                        first_action[nxt] = fa
                        heapq.heappush(pq, (nd, nxt))
            else:
                e = lval
                i = self.eidx[e]
                floor = ftuple[i]
                nd = cur_d + enter_exit_cost
                a = f"EXIT{{{p},{e}}}"
                fa = a if first_action[node] is None else first_action[node]
                if floor == goal:
                    if best_terminal_cost is None or nd < best_terminal_cost:
                        best_terminal_cost = nd
                        best_terminal_action = fa
                else:
                    nxt = (ftuple, "floor", floor)
                    if nd < dist.get(nxt, float("inf")):
                        dist[nxt] = nd
                        first_action[nxt] = fa
                        heapq.heappush(pq, (nd, nxt))

        if best_terminal_action is None:
            result = (float("inf"), None)
        else:
            result = (best_terminal_cost, best_terminal_action)
        if len(self._path_cache) > 30000:
            self._path_cache.clear()
        self._path_cache[key] = result
        return result

    def _learn_from_last_step(self, current_state):
        if self.prev_state is None or self.prev_action is None:
            return
        a = self.prev_action
        before = self.prev_state
        after = current_state
        gained = float(self.game.get_last_gained_reward())

        if a == "RESET":
            return

        kind, x, y = self._parse_action(a)
        if kind == "MOVE":
            e, trg = x, y
            b = dict((eid, fl) for eid, fl, _ in before[0])[e]
            c = dict((eid, fl) for eid, fl, _ in after[0])[e]
            success = (b == trg and c == trg) or (b != trg and c == trg)
            if success:
                self.move_stats[e][0] += 1.0
            else:
                self.move_stats[e][1] += 1.0
        elif kind == "ENTER":
            p = x
            b = dict(before[1]).get(p)
            c = dict(after[1]).get(p)
            success = (b is not None and b[0] == "floor" and c == ("in", y))
            if success:
                self.person_stats[p][0] += 1.0
            else:
                self.person_stats[p][1] += 1.0
        elif kind == "EXIT":
            p = x
            b = dict(before[1]).get(p)
            c = dict(after[1]).get(p)
            success = (b == ("in", y) and c != ("in", y))
            if success:
                self.person_stats[p][0] += 1.0
            else:
                self.person_stats[p][1] += 1.0
            if success and gained > 0.0 and b == ("in", y):
                floor_before = dict((eid, fl) for eid, fl, _ in before[0])[y]
                if floor_before == self.person_goal[p]:
                    reward_part = gained
                    if before[2] == 1:
                        reward_part -= self.goal_reward
                    if reward_part > 0.0:
                        self.reward_sum[p] += reward_part
                        self.reward_n[p] += 1

    def _legal_actions(self, state):
        elevators_t, persons_t, _ = state
        actions = ["RESET"]
        elev_floor = {e: f for e, f, _ in elevators_t}
        elev_load = {e: w for e, _, w in elevators_t}
        persons_map = dict(persons_t)

        for e in self.elev_ids:
            cur = elev_floor[e]
            for f in self.reachable[e]:
                actions.append(f"MOVE{{{e},{f}}}")
            for p, loc in persons_map.items():
                if loc == ("floor", cur):
                    if elev_load[e] + self.person_weight[p] <= self.capacities[e]:
                        actions.append(f"ENTER{{{p},{e}}}")
                if loc == ("in", e):
                    actions.append(f"EXIT{{{p},{e}}}")
        return actions

    @staticmethod
    def _parse_action(a):
        if a == "RESET":
            return ("RESET", 0, 0)
        n = a.index("{")
        m = a.index(",")
        k = a.index("}")
        return (a[:n], int(a[n + 1:m]), int(a[m + 1:k]))
