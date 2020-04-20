"""
Code related to applying a mask onto a variable to impose kernel sparsity,
aka model pruning, on a TensorFlow graph.
"""

from typing import Tuple, List, Union
from collections import namedtuple
import tensorflow.contrib.graph_editor as ge

from neuralmagicML.tensorflow.utils import (
    tf_compat,
    tf_compat_div,
    clean_tensor_name,
    get_op_input_var,
    get_tensor_var,
    eval_tensor_sparsity,
    non_zero_mask_initializer,
)


__all__ = [
    "PruningOpVars",
    "KSScope",
    "create_op_pruning",
    "create_graph_ops_pruning",
    "get_or_create_graph_ops_pruning",
    "apply_op_vars_masks",
    "create_summaries_pruning",
    "create_ks_schedule_ops",
    "get_or_create_ks_schedule_ops",
    "get_or_create_ks_scheduled_graph_ops",
]


PruningOpVars = namedtuple(
    "PruningOpVars", ["op", "op_input", "update", "mask", "masked"]
)


class KSScope(object):
    """
    Convenience class for dealing with scope and names for kernel sparsity
    in the tf graph.
    """

    NM_KS = "nm_ks"
    NM_KS_OPS = "nm_ks_ops"

    OPS_UPDATE = "update_ops"
    OPS_SUMMARY = "summary_ops"
    OPS_SCHEDULE = "schedule_ops"
    OPS_SPARSITY = "sparsity_ops"

    OP_COND_UPDATE = "nm_conditional_update"
    OP_SPARSITY = "nm_sparsity"
    OP_UPDATE_READY = "nm_update_ready"
    OP_MASKED_VAR = "nm_masked_var"
    OP_MASK_ASSIGN = "nm_mask_assign"
    OP_PRUNE_VARS_ASSIGN = "nm_prune_vars_assign"
    OP_MASK_UPDATE_NO_OP = "nm_mask_update_no_op"
    OP_MASK_UPDATE = "nm_mask_update"
    OP_SAVE = "nm_save"

    VAR_MASK = "nm_mask"
    VAR_THRESHOLD = "nm_threshold"

    @staticmethod
    def general(ks_group: str, additional: str = None, trailing_slash: bool = False):
        """
        Create a general kernel sparsity scope in the tf graph.
        Use cases are for generic ops like target sparsity, conditional updates, etc.

        :param ks_group: the group identifier the scope should be created under
        :param additional: any additional scope that should be added to the end
        :param trailing_slash: include a trailing forward slash if True, else False
        :return: the proper scope
        """
        scope = KSScope._format(KSScope.NM_KS_OPS, ks_group)
        scope = KSScope._format(
            scope, additional=additional, trailing_slash=trailing_slash
        )

        return scope

    @staticmethod
    def model(
        op_tens: tf_compat.Tensor,
        ks_group: str,
        additional: str = None,
        trailing_slash: bool = False,
    ) -> str:
        """
        Create a model specific kernel sparsity scope in the tf graph.
        Use cases are for the specific mask, threshold, etc variables
        to induce sparsity along with the ops to update those vars.

        :param op_tens: the op tensor to create the scope for
        :param ks_group: the group identifier the scope should be created under
        :param additional: any additional scope that should be added to the end
        :param trailing_slash: include a trailing forward slash if True, else False
        :return: the proper scope
        """
        op_name = clean_tensor_name(op_tens)
        scope = KSScope._format("{}_{}".format(op_name, KSScope.NM_KS), ks_group)
        scope = KSScope._format(
            scope, additional=additional, trailing_slash=trailing_slash
        )

        return scope

    @staticmethod
    def collection_name(ks_group: str, name: str) -> str:
        """
        Create a predictable name for a given variable / op in a group for lookup /
        storage in a collection

        :param ks_group: the group identifier the name belongs under
        :param name: the name of the op or variable to be stored or retrieved
        :return: the formatted name for use in a collection
        """
        return "nm_ks_collection_{}_{}".format(ks_group, name)

    @staticmethod
    def _format(
        current: str, additional: str = None, trailing_slash: bool = False
    ) -> str:
        scope = current

        if additional is not None:
            scope = "{}/{}".format(current, additional)

        if trailing_slash:
            scope += "/"

        return scope


def create_op_pruning(
    op: tf_compat.Operation,
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    update_ready: tf_compat.Tensor,
    ks_group: str,
) -> PruningOpVars:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to an operators variable.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param op: the operation to prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param update_ready: the tensor where if true will update the mask from sparsity,
        if false will not update the mask
    :param ks_group: the group identifier the scope should be created under
    :return: a named tuple containing the assignment op, mask variable,
        threshold tensor, and masked tensor
    """
    op_sgv = ge.sgv(op)
    op_var_tens = get_op_input_var(op, var_index)

    # create the necessary variables first
    with tf_compat.variable_scope(
        KSScope.model(op, ks_group), reuse=tf_compat.AUTO_REUSE
    ):
        mask = tf_compat.get_variable(
            KSScope.VAR_MASK,
            op_var_tens.get_shape(),
            initializer=non_zero_mask_initializer(op_var_tens),
            trainable=False,
            dtype=op_var_tens.dtype,
        )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_MASK), mask
    )

    # create the masked operation and assign as the new input to the op
    with tf_compat.name_scope(KSScope.model(op, ks_group, trailing_slash=True)):
        masked = tf_compat.math.multiply(mask, op_var_tens, KSScope.OP_MASKED_VAR)
        op_swapped_inputs = [
            inp if inp != op_var_tens else masked for inp in op_sgv.inputs
        ]
        ge.swap_inputs(op, op_swapped_inputs)
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASKED_VAR), masked
    )

    def _update():
        # create the update ops using the target sparsity tensor
        with tf_compat.name_scope(
            KSScope.model(
                op, ks_group, additional=KSScope.OPS_UPDATE, trailing_slash=True,
            )
        ):
            abs_var = tf_compat.abs(op_var_tens)
            sparse_index = tf_compat.cast(
                tf_compat.math.round(
                    tf_compat.cast(tf_compat.size(abs_var), tf_compat.dtypes.float32)
                    * (1.0 - sparsity)
                ),
                tf_compat.dtypes.int32,
            )
            sparse_index = tf_compat.minimum(
                tf_compat.maximum(sparse_index, 0), tf_compat.size(op_var_tens) - 1
            )
            sorted_vals, _ = tf_compat.math.top_k(
                tf_compat.reshape(abs_var, [-1]), k=tf_compat.size(abs_var)
            )
            threshold = tf_compat.gather(
                sorted_vals, sparse_index, name=KSScope.VAR_THRESHOLD
            )
            new_mask = tf_compat.cast(
                tf_compat.greater(abs_var, threshold), tf_compat.dtypes.float32
            )

            return tf_compat.assign(mask, new_mask, name=KSScope.OP_MASK_ASSIGN)

    def _no_update():
        with tf_compat.name_scope(
            KSScope.model(
                op, ks_group, additional=KSScope.OPS_UPDATE, trailing_slash=True,
            )
        ):
            return tf_compat.constant(
                0.0, dtype=op_var_tens.dtype, name=KSScope.OP_MASK_UPDATE_NO_OP
            )

    with tf_compat.name_scope(
        KSScope.model(op, ks_group, additional=KSScope.OPS_UPDATE, trailing_slash=True,)
    ):
        mask_update = tf_compat.cond(
            update_ready, _update, _no_update, name=KSScope.OP_MASK_UPDATE
        )

    # add return state to collections
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_UPDATE), mask_update
    )

    return PruningOpVars(op, op_var_tens, mask_update, mask, masked)


def create_graph_ops_pruning(
    graph: tf_compat.Graph,
    op_names: List[str],
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    update_ready: tf_compat.Tensor,
    ks_group: str,
) -> List[PruningOpVars]:
    """
    Creates the necessary variables and operators to gradually
    apply sparsity to a given list of operators in a graph.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param op_names: the list of name of the operations in the
        graph to prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param update_ready: the tensor where if true will update the mask from sparsity,
        if false will not update the mask
    :param ks_group: the group identifier the scope should be created under
    :return: a list of the created named tuples each containing the
        assignment op, mask variable, threshold tensor, and masked tensor
    """
    pruning_op_vars = []

    for op_name in op_names:
        op = graph.get_operation_by_name(op_name)
        op_vars = create_op_pruning(op, var_index, sparsity, update_ready, ks_group)
        pruning_op_vars.append(op_vars)

    return pruning_op_vars


def get_or_create_graph_ops_pruning(
    graph: tf_compat.Graph,
    op_names: List[str],
    var_index: Union[int, str],
    sparsity: tf_compat.Tensor,
    update_ready: tf_compat.Tensor,
    ks_group: str,
) -> List[PruningOpVars]:
    """
    Creates or retrieves (if previously created) the necessary variables
    and operators to gradually apply sparsity to a given list of operators in a graph.

    Handles setting a mask on an operator to the given sparsity.
    Sets the mask based on pruning away the lowest absolute magnitude weights.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param op_names: the list of name of the operations in the graph to
        prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param sparsity: the target sparsity to use for assigning the masks
    :param update_ready: the tensor where if true will update the mask from sparsity,
        if false will not update the mask
    :param ks_group: the group identifier the scope should be created under
    :return: a list of the created or retrieved named tuples each containing the
        assignment op, mask variable, threshold tensor, and masked tensor
    """
    mask_updates = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASK_UPDATE)
    )
    masks = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.VAR_MASK)
    )
    maskeds = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_MASKED_VAR)
    )

    if len(mask_updates) < 1 or len(masks) < 1 or len(maskeds) < 1:
        pruning_op_vars = create_graph_ops_pruning(
            graph, op_names, var_index, sparsity, update_ready, ks_group
        )
    else:
        pruning_op_vars = []
        ops = [graph.get_operation_by_name(op_name) for op_name in op_names]
        op_inps = [get_op_input_var(op, var_index) for op in ops]

        for op, op_inp, mask_update, mask, masked in zip(
            ops, op_inps, mask_updates, masks, maskeds
        ):
            pruning_op_vars.append(PruningOpVars(op, op_inp, mask_update, mask, masked))

    return pruning_op_vars


def create_summaries_pruning(pruning_op_vars: List[PruningOpVars]):
    """
    Create TensorBoard summary ops in the current graph for the
    given list of PruningOpVars.

    :param pruning_op_vars: the list of named tuples containing the masked input to the
        pruned op to record sparsity for in TensorBoard.
    :return: the created summaries for the pruned op vars
    """
    summaries = []

    for op_vars in pruning_op_vars:
        sum_op = tf_compat.summary.scalar(
            "Modifier KS/{}".format(clean_tensor_name(op_vars.op)),
            tf_compat.math.zero_fraction(op_vars.masked),
        )
        summaries.append(sum_op)

    return summaries


def apply_op_vars_masks(
    pruning_op_vars: List[PruningOpVars], ks_group: str, sess: tf_compat.Session
):
    """
    Apply the masks to the original ops input var so that it can be saved
    with the desired sparsity for later.

    :param pruning_op_vars: the list of named tuples containing the sparse mask
        and the op variable to apply the sparse mask to
    :param ks_group: the group to create the assign ops under
    :param sess: the session to use to run the assign
    """
    for op_vars in pruning_op_vars:
        with tf_compat.name_scope(KSScope.model(op_vars.op, ks_group, KSScope.OP_SAVE)):
            masked_var = tf_compat.math.multiply(op_vars.op_input, op_vars.mask)
            input_var = get_tensor_var(op_vars.op_input)
            assign = tf_compat.assign(input_var, masked_var)
            sess.run(assign)


def create_ks_schedule_ops(
    global_step: tf_compat.Variable,
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    Create a gradual schedule for model pruning (kernel sparsity).
    Creates a sparsity tensor that goes from init_sparsity til final_sparsity
    starting at begin_step and ending at end_step.
    Uses the global_step to map those.
    Additionally creates an update_ready tensor that is True if an update
    to the sparsity tensor should be run, False otherwise.

    :param global_step: the global optimizer step for the training graph
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a
        weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between
        init_sparsity and final_sparsity higher values will lead to larger sparsity
        steps at the beginning vs the end ie: linear (1) vs cubic (3)
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the signal for update_ready and the target sparsity
    """

    # create the scheduling ops first and the sparsity ops
    with tf_compat.name_scope(
        KSScope.general(ks_group, additional=KSScope.OPS_SCHEDULE, trailing_slash=True)
    ):
        sched_before = tf_compat.less(global_step, begin_step)
        sched_start = tf_compat.equal(global_step, begin_step)
        sched_end = tf_compat.equal(global_step, end_step)
        sched_active = tf_compat.logical_and(
            tf_compat.greater(global_step, begin_step),
            tf_compat.less(global_step, end_step),
        )
        sched_active_inclusive = tf_compat.logical_or(
            sched_active, tf_compat.logical_or(sched_start, sched_end)
        )
        sched_update = tf_compat.cond(
            tf_compat.less_equal(update_step_freq, 0),
            lambda: tf_compat.constant(True),
            lambda: tf_compat.equal(
                tf_compat.mod((global_step - begin_step), update_step_freq), 0
            ),
        )
        sched_update_ready = tf_compat.logical_or(
            tf_compat.logical_or(sched_start, sched_end), sched_update
        )

        percentage = tf_compat.minimum(
            1.0,
            tf_compat.maximum(
                0.0,
                tf_compat_div(
                    tf_compat.cast(global_step - begin_step, tf_compat.dtypes.float32),
                    end_step - begin_step,
                ),
            ),
        )
        exp_percentage = tf_compat.pow(percentage, 1 / exponent)
        calc_sparsity = (
            tf_compat.multiply(final_sparsity - init_sparsity, exp_percentage)
            + init_sparsity
        )

    # create the update ready tensor and sparsity tensor
    with tf_compat.name_scope(KSScope.general(ks_group, trailing_slash=True)):
        update_ready = tf_compat.logical_and(
            sched_active_inclusive, sched_update_ready, name=KSScope.OP_UPDATE_READY,
        )
        sparsity = tf_compat.case(
            [
                (sched_before, lambda: tf_compat.constant(0.0)),
                (sched_start, lambda: tf_compat.constant(init_sparsity)),
                (sched_active, lambda: calc_sparsity),
            ],
            default=lambda: tf_compat.constant(final_sparsity),
            name=KSScope.OP_SPARSITY,
        )

    # add return state to collections
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_UPDATE_READY), update_ready
    )
    tf_compat.add_to_collection(
        KSScope.collection_name(ks_group, KSScope.OP_SPARSITY), sparsity
    )

    return update_ready, sparsity


def get_or_create_ks_schedule_ops(
    global_step: tf_compat.Tensor,
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, tf_compat.Tensor]:
    """
    Creates or retrieves (if previously created) a gradual schedule
    for model pruning (kernel sparsity).
    Creates a sparsity tensor that goes from init_sparsity til final_sparsity
    starting at begin_step and ending at end_step.
    Uses the global_step to map those.
    Additionally creates an update_ready tensor that is True if an update
    to the sparsity tensor should be run, False otherwise.

    :param global_step: the global optimizer step for the training graph
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a
        weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between
        init_sparsity and final_sparsity higher values will lead to larger sparsity
        steps at the beginning vs the end ie: linear (1) vs cubic (3)
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the signal for update_ready and the target sparsity
    """
    update_ready = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_UPDATE_READY)
    )
    sparsity = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_SPARSITY)
    )

    update_ready = update_ready[0] if len(update_ready) > 0 else None
    sparsity = sparsity[0] if len(sparsity) > 0 else None

    if update_ready is None or sparsity is None:
        update_ready, sparsity = create_ks_schedule_ops(
            global_step,
            begin_step,
            end_step,
            update_step_freq,
            init_sparsity,
            final_sparsity,
            exponent,
            ks_group,
        )

    return update_ready, sparsity


def get_or_create_ks_scheduled_graph_ops(
    graph: tf_compat.Graph,
    global_step: tf_compat.Variable,
    op_names: List[str],
    var_index: Union[int, str],
    begin_step: int,
    end_step: int,
    update_step_freq: int,
    init_sparsity: float,
    final_sparsity: float,
    exponent: float,
    ks_group: str,
) -> Tuple[tf_compat.Tensor, List[PruningOpVars], tf_compat.Tensor, tf_compat.Tensor]:
    """
    Gets or creates model pruning (kernel sparsity) ops and vars in the graph
    to be applied over a specific schedule.
    Creates them for the op_names in the graph such that they follow a schedule
    from begin_step to end_step starting at init_sparsity and ending at final_sparsity.

    :param graph: the tf graph to pull the operator out of for applying the pruning to
    :param global_step: the global optimizer step for the training graph
    :param op_names: the list of name of the operations in the graph to
        prune to the given sparsity
    :param var_index: the index for where the variable is,
        see :py:func:`~get_op_input_var`
    :param begin_step: the global step to begin pruning at
    :param end_step: the global step to end pruning at
    :param update_step_freq: the number of global steps between each weight update
    :param init_sparsity: the starting value for sparsity of a
        weight tensor to be enforce
    :param final_sparsity: the end value for sparsity for a weight tensor to be enforce
    :param exponent: the exponent to use for interpolating between
        init_sparsity and final_sparsity higher values will lead to larger sparsity
        steps at the beginning vs the end ie: linear (1) vs cubic (3)
    :param ks_group: the group identifier the scope should be created under
    :return: a tuple containing the update operation to run in a session,
        a list of the pruning ops and vars for each desired op in the graph,
        the tensor containing the update_ready signal for the pruning ops,
        the tensor containing the set sparsity for the pruning ops
    """
    update_ready, sparsity = get_or_create_ks_schedule_ops(
        global_step,
        begin_step,
        end_step,
        update_step_freq,
        init_sparsity,
        final_sparsity,
        exponent,
        ks_group,
    )
    pruning_op_vars = get_or_create_graph_ops_pruning(
        graph, op_names, var_index, sparsity, update_ready, ks_group
    )

    update_op = tf_compat.get_collection(
        KSScope.collection_name(ks_group, KSScope.OP_COND_UPDATE)
    )
    update_op = update_op[0] if len(update_op) > 0 else None

    if update_op is None:
        update_op = tf_compat.group(*[op_var.update for op_var in pruning_op_vars])

        # add return state to collections
        tf_compat.add_to_collection(
            KSScope.collection_name(ks_group, KSScope.OP_COND_UPDATE), update_op
        )

    return update_op, pruning_op_vars, update_ready, sparsity
