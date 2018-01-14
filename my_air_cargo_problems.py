from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:
                        #for all cargo and planes in airport
                        precond_pos = [expr("At({}, {})".format(c, a)),
                                       expr("At({}, {})".format(p, a))]
                        precond_neg = []
                        #add cargo to plane
                        effect_add = [expr("In({}, {})".format(c, p))]
                        #remove cargo from airport
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                      [precond_pos, precond_neg],
                                      [effect_add, effect_rem])
                        loads.append(load)
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:
                        # for all cargo in all planes in all airports
                        precond_pos = [expr("In({}, {})".format(c, p)),
                                       expr("At({}, {})".format(p, a))]
                        precond_neg = []
                        #add cargo to airport
                        effect_add = [expr("At({}, {})".format(c, a))]
                        #remove cargo from plane
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        unloads.append(unload)
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            #for all planes in origin
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            #add plane to destination
                            effect_add = [expr("At({}, {})".format(p, to))]
                            #remove plane from origin
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param active_state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # get active active_state
        active_state = decode_state(state, self.state_map)
        # Declare possible_actions variable
        possible_actions = []
        # get the action objects (load, unload, and fly)
        action_objs = self.get_actions()
        # for all actions, check if preconditions are met
        for action in action_objs:
                #if fluents of preconds are in state
            if all([precs in active_state.pos for precs in action.precond_pos]):
                # add actions to possibles list.
                possible_actions.append(action)

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # get active state
        active_state = decode_state(state, self.state_map)
        # remove atoms from state , if they are in effect_rem list.
        active_state.pos = [f for f in active_state.pos if f not in action.effect_rem]
        # add fluents from effect_add to state
        active_state.pos += action.effect_add
        # return next state
        return encode_state(active_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # Find all goals that have not been achieved and return count
        state = decode_state(node.state, self.state_map)
        remaining_goals = [g for g in self.goal if g not in state.pos]
        result = len(remaining_goals)

        return result


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # implement Problem 2 definition
    cargos = ["C1", "C2", "C3"]
    planes = ["P1", "P2", "P3"]
    airports = ["JFK", "SFO", "ATL"]
    #in initial state (cargos, airports, planes)
    pos = [expr("At(C1, SFO)"),
           expr("At(C2, JFK)"),
           expr("At(C3, ATL)"),
           expr("At(P1, SFO)"),
           expr("At(P2, JFK)"),
           expr("At(P3, ATL)")
           ]
    #not in inital state (cargos, airports, planes)
    neg = [expr("At(C1, JFK)"), expr("At(C1, ATL)"), expr("At(C2, SFO)"), expr("At(C2, ATL)"),
           expr("At(C3, JFK)"), expr("At(C3, SFO)"), expr("At(P1, JFK)"), expr("At(P1, ATL)"),
           expr("At(P2, SFO)"), expr("At(P2, ATL)"), expr("At(P3, JFK)"), expr("At(P3, SFO)"),
           expr("In(C1, P1)"), expr("In(C1, P2)"), expr("In(C1, P3)"),
           expr("In(C2, P1)"), expr("In(C2, P2)"), expr("In(C2, P3)"),
           expr("In(C3, P1)"), expr("In(C3, P2)"), expr("In(C3, P3)")
           ]
    #full state
    init = FluentState(pos, neg)
    #goal p2
    goal = [expr("At(C1, JFK)"),
            expr("At(C2, SFO)"),
            expr("At(C3, SFO)")
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # implement Problem 3 definition
    cargos = ["C1", "C2", "C3", "C4"]
    planes = ["P1", "P2"]
    airports = ["JFK", "SFO", "ATL", "ORD"]
    # in initial state (cargos, airports, planes)
    pos = [expr("At(C1, SFO)"),
           expr("At(C2, JFK)"),
           expr("At(C3, ATL)"),
           expr("At(C4, ORD)"),
           expr("At(P1, SFO)"),
           expr("At(P2, JFK)")
           ]
    # not in inital state (cargos, airports, planes)
    neg = [expr("At(C1, JFK)"), expr("At(C1, ATL)"), expr("At(C1, ORD)"),
           expr("At(C2, SFO)"), expr("At(C2, ATL)"), expr("At(C2, ORD)"),
           expr("At(C3, JFK)"), expr("At(C3, SFO)"), expr("At(C3, ORD)"),
           expr("At(C4, JFK)"), expr("At(C4, SFO)"), expr("At(C4, ATL)"),
           expr("At(P1, JFK)"), expr("At(P1, ATL)"), expr("At(P1, ORD)"),
           expr("At(P2, SFO)"), expr("At(P2, ATL)"), expr("At(P2, ORD)"),
           expr("In(C1, P1)"), expr("In(C1, P2)"),
           expr("In(C2, P1)"), expr("In(C2, P2)"),
           expr("In(C3, P1)"), expr("In(C3, P2)"),
           expr("In(C4, P1)"), expr("In(C4, P2)")
           ]
    #full state
    init = FluentState(pos, neg)
    #goal p3
    goal = [expr("At(C1, JFK)"),
            expr("At(C3, JFK)"),
            expr("At(C2, SFO)"),
            expr("At(C4, SFO)"),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)