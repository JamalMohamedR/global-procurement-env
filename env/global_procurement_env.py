# global_procurement_env.py — The main environment class wrapping all Day 1 modules.
# Implements the three OpenEnv-required methods: reset(), step(), state().
# All three return typed Pydantic models, NOT plain dicts.

import random
import json
import os
import copy
from api.schemas import Observation, Action, Reward, SupplierObservation
from env.supply_chain_sim import SupplyChainState, apply_action
from env.constraint_engine import load_policy, validate_action
from env.disruption_engine import DisruptionEngine
from env.tasks import task1_easy, task2_medium, task3_hard
from graders import grader1, grader2, grader3

# Map task number to its grader module
GRADER_MAP = {1: grader1, 2: grader2, 3: grader3}


# Map task number to its config module
TASK_CONFIGS = {
    1: task1_easy,
    2: task2_medium,
    3: task3_hard,
}

# Path to the suppliers JSON data file
SUPPLIERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "suppliers.json")


class GlobalProcurementEnv:
    def __init__(self):
        self._state: SupplyChainState | None = None
        self.task: int = 1
        self.seed: int = 42
        self.disruption_engine: DisruptionEngine | None = None
        self.is_done: bool = False
        self.current_task: int = 1
        self._task_config = None
        self._active_disruptions: list = []
        # Episode log — rebuilt on every reset(), consumed by graders at done=True
        self._episode_log: dict = {}
        self._grader_score: float | None = None

    def reset(self, task: int = 1, seed: int = 42) -> Observation:
        """
        Returns: Observation (Pydantic model) -- NOT a dict.
        CRITICAL: Always self._state = SupplyChainState(...) -- never .clear() a list.
        """
        random.seed(seed)  # Must be first -- makes every run with same seed identical

        self.task = task
        self.current_task = task
        self.seed = seed
        self.is_done = False
        self._active_disruptions = []
        self._grader_score = None
        # Fresh episode log — never mutate the old one (Rule 2)
        self._episode_log = {
            "steps": [],
            "total_violations": 0,
            "total_lead_days": 0,
            "total_carbon": 0.0,
            "final_budget": 200000.0,
        }

        # Load task config
        self._task_config = TASK_CONFIGS.get(task, task1_easy)

        # CRITICAL: Always create a FRESH state object -- never mutate the old one
        self._state = SupplyChainState(
            budget=200000.0,
            suppliers=self._load_suppliers_for_task(task),
            inventory={"steel": 0.0, "chips": 0.0, "fabric": 0.0},
            lead_days=0,
            carbon=0.0,
            step_count=0,
            violations=0,
        )

        self.disruption_engine = DisruptionEngine(task=task, seed=seed)
        return self._build_observation()

    def step(self, action: int) -> tuple:
        """
        Returns: (Observation, Reward, bool, dict) -- a 4-tuple per OpenEnv spec.
        NOT a dict. The bool is done. The dict is info.
        """
        if self.is_done:
            obs = self._build_observation()
            reward = Reward(value=0.0, compliance=0.0, cost_efficiency=0.0,
                            delivery_speed=0.0, carbon_score=0.0)
            return obs, reward, True, {"error": "episode_already_done"}

        # Step 1: disruption check
        active = self.disruption_engine.check(self._state.step_count)
        self._active_disruptions = [d["name"] for d in active]
        available_suppliers = self._apply_disruptions(active)

        # Step 2: apply action
        result = apply_action(self._state, action, available_suppliers)

        # Step 3: validate against policy
        selected = result.get("selected_supplier")
        if selected:
            policy = load_policy(selected["country"])
            is_valid, soft_penalty, vtype = validate_action(
                self._state, selected, policy
            )
        else:
            is_valid = True
            soft_penalty = 0.0
            vtype = None

        # Step 4: hard violation -> episode ends immediately
        if not is_valid:
            self._state.violations += 1
            self.is_done = True
            # Log the step before grading
            self._episode_log["steps"].append(self._state.step_count)
            self._episode_log["total_violations"] = self._state.violations
            self._episode_log["total_lead_days"] += self._state.lead_days
            self._episode_log["total_carbon"] = self._state.carbon
            self._episode_log["final_budget"] = self._state.budget
            grader_module = GRADER_MAP.get(self.task, grader1)
            self._grader_score = grader_module.grade(self._episode_log)
            obs = self._build_observation()
            reward = Reward(
                value=-1.0, compliance=0.0, cost_efficiency=0.0,
                delivery_speed=0.0, carbon_score=0.0
            )
            return obs, reward, True, {"violation": vtype}

        # Step 5: compute reward
        reward_obj = self._compute_reward(soft_penalty)
        done = self._state.step_count >= self._task_config.MAX_STEPS
        self.is_done = done

        # Track episode log for grader
        self._episode_log["steps"].append(self._state.step_count)
        self._episode_log["total_violations"] = self._state.violations
        self._episode_log["total_lead_days"] += self._state.lead_days
        self._episode_log["total_carbon"] = self._state.carbon
        self._episode_log["final_budget"] = self._state.budget

        # Call the appropriate grader when the episode ends
        if done:
            grader_module = GRADER_MAP.get(self.task, grader1)
            self._grader_score = grader_module.grade(self._episode_log)

        return self._build_observation(), reward_obj, done, {}

    def state(self) -> Observation:
        """Returns current state as Pydantic Observation -- does not advance."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._build_observation()

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Converts internal SupplyChainState into a Pydantic Observation."""
        supplier_obs = [
            SupplierObservation(
                id=s["id"],
                country=s["country"],
                price_usd=float(s["price_usd"]),
                lead_days=int(s["lead_days"]),
                carbon_tons=float(s["carbon_tons"]),
                available=bool(s["available"]),
                applied_duty_rate=float(s.get("applied_duty_rate", 0.0)),
            )
            for s in self._state.suppliers
        ]
        return Observation(
            step=self._state.step_count,
            budget_remaining=self._state.budget,
            inventory=dict(self._state.inventory),
            suppliers=supplier_obs,
            active_disruptions=list(self._active_disruptions),
            policy_violations_this_episode=self._state.violations,
            current_task=self.task,
            grader_score=self._grader_score,
        )

    def _load_suppliers_for_task(self, task: int) -> list:
        """Loads suppliers from JSON and filters by task-allowed countries."""
        with open(SUPPLIERS_PATH, "r") as f:
            all_suppliers = json.load(f)

        task_config = TASK_CONFIGS.get(task, task1_easy)
        allowed_countries = task_config.SUPPLIER_COUNTRIES

        return [s for s in all_suppliers if s["country"] in allowed_countries]

    def _apply_disruptions(self, active_disruptions: list) -> list:
        """
        Takes a list of active disruption dicts and modifies supplier availability
        or lead times accordingly. Returns the modified supplier list.
        """
        suppliers = copy.deepcopy(self._state.suppliers)

        for disruption in active_disruptions:
            affected_ids = disruption.get("affected_suppliers", [])
            effect = disruption.get("effect", "")

            for supplier in suppliers:
                if supplier["id"] in affected_ids:
                    if effect == "unavailable":
                        supplier["available"] = False
                    elif effect == "lead_time_multiplier":
                        supplier["lead_days"] = int(
                            supplier["lead_days"] * disruption.get("multiplier", 1.0)
                        )
                    elif effect == "lead_time_add":
                        supplier["lead_days"] += disruption.get("days_added", 0)

        return suppliers

    def _compute_reward(self, soft_penalty: float = 0.0) -> Reward:
        """
        Reward = 0.35 x compliance + 0.25 x cost + 0.25 x speed + 0.15 x carbon
        Each component is normalised to [0, 1].
        Returns a Pydantic Reward model.
        """
        MAX_BUDGET = 200000.0
        MAX_LEAD_DAYS = 30
        MAX_CARBON = 50.0
        MAX_VIOLATIONS = 10

        compliance = max(0.0, 1.0 - (self._state.violations / MAX_VIOLATIONS))
        cost_efficiency = max(0.0, self._state.budget / MAX_BUDGET)
        delivery_speed = max(0.0, 1.0 - (self._state.lead_days / MAX_LEAD_DAYS))
        carbon_score = max(0.0, 1.0 - (self._state.carbon / MAX_CARBON))

        value = (
            0.35 * compliance
            + 0.25 * cost_efficiency
            + 0.25 * delivery_speed
            + 0.15 * carbon_score
        ) + soft_penalty

        value = round(max(-1.0, min(1.0, value)), 4)

        return Reward(
            value=value,
            compliance=round(compliance, 4),
            cost_efficiency=round(cost_efficiency, 4),
            delivery_speed=round(delivery_speed, 4),
            carbon_score=round(carbon_score, 4),
        )

    def _get_max_steps(self) -> int:
        """Returns the max steps for the current task."""
        return self._task_config.MAX_STEPS