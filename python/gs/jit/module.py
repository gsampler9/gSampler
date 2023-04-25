from typing import List
from .trace import gs_symbolic_trace
from ..matrix_api import Matrix
from .passes import dce, cse
from .optimize import merge_relabel_and_all_indices

CONVERT_2_MATRIX = "Convert2Matrix"
STATIS_LIST = "StatisList"
GRAPH_ARG = "GRAPH_ARG"
STATIC_ARG = "STATIC_ARG"


def get_actions(args):
    actions = []
    graph_args_count = 0
    static_args_count = 0
    for arg_offset, arg in enumerate(args):
        if isinstance(arg, Matrix):
            actions.append(
                (GRAPH_ARG, graph_args_count, arg_offset, CONVERT_2_MATRIX))
            graph_args_count += 1
        elif isinstance(arg, List):
            actions.append(
                (STATIC_ARG, static_args_count, arg_offset, STATIS_LIST))
            static_args_count += 1
        else:
            actions.append(None)
    return actions


def split_actions(actions):
    graph_actions = []
    static_actions = []
    for action in actions:
        if action is None:
            continue

        if action[0] == GRAPH_ARG:
            graph_actions.append(action)
        elif action[0] == STATIC_ARG:
            static_actions.append(action)

    return graph_actions, static_actions


def generate_graph_args(args, graph_actions):
    graph_args = []
    for action in graph_actions:
        _, _, arg_offset, a = action
        if a == CONVERT_2_MATRIX:
            graph_args.append(args[arg_offset]._graph)
        else:
            raise ValueError
    return graph_args


def generate_static_args(args, static_actions):
    static_args = []
    for action in static_actions:
        _, _, arg_offset, a = action
        if a == STATIS_LIST:
            static_args.append(args[arg_offset])
        else:
            raise ValueError
    return static_args


def generate_new_args(args, graph_args, static_args, actions):
    new_args = []
    for index, action in enumerate(actions):
        if action is None:
            new_args.append(args[index])
        else:
            _, offset, _, a = action
            if a == CONVERT_2_MATRIX:
                new_args.append(Matrix(graph_args[offset]))
            elif a == STATIS_LIST:
                new_args.append(static_args[offset])
            else:
                raise ValueError
    return tuple(new_args)


class compile:

    def __init__(self, func, args):
        '''
        This is auto wrapper for user's func.
        We will create an func inner_wrapper according user's func and its args.
        Each arg in args will have an action, which is used to tell what we should do for the arg.

        There are three types action, 'None', 'GRAPH_ARG', 'STATIC_ARG'.
        We use actions to generate graph_args, static_args.
        And in inner_wrapper, we will leverage graph_args, static_args and user's args with actions to generate new_args.
        By this way, we can convert some args into static args (e.g. fanout=[5,10]) and expand Matrix's methods.

        For 'None' action, we will do nothing and pass arg through into inner_wrapper.

        For 'GRAPH_ARG' action, where arg is 'Matrix', we will store Matrix._graph in graph_args
        and in inner_wrapper, we will use Matrix._graph in graph_args to re-generate origin Matrix.

        For 'STATIC_ARG' action, where arg will not change during training (e.g. fanout, metapath or others). We will store them
        in static_args and they will appear as constants in torch.fx.
        '''

        # generate actions
        actions = get_actions(args)
        # extract graph_actions and static_actions from acitons
        graph_actions, static_actions = split_actions(actions)
        self.graph_actions = graph_actions
        self.static_actions = static_actions
        # generate static_args via static_actions
        static_args = generate_static_args(args, self.static_actions)

        def inner_wrapper(inner_args, inner_graph_args):
            # generate new_args for user's arg.
            # arg in static_args will be compiled as contants.
            # arg in graph_args will be leveraged to generate Matrix.
            new_args = generate_new_args(inner_args, inner_graph_args,
                                         static_args, actions)
            return func(*new_args)

        # compiled to torch.fx IR
        gm = gs_symbolic_trace(inner_wrapper)

        # optimization
        gm = merge_relabel_and_all_indices(gm)

        # pass
        gm = dce(gm)

        self.gm = gm

    def __call__(self, *args):
        # generate graph_actions via graph_actions
        # In fact, it stores *._graph in graph_args
        graph_args = generate_graph_args(args, self.graph_actions)
        return self.gm(args, graph_args)