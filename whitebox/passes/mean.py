import operator

import torch
from torch import fx

# Import the attention layer module to ensure the custom op is registered
from vllm.attention.layer import Attention  # noqa: F401
from vllm.compilation.inductor_pass import InductorPass


def capture_mean(mean_tensor, layer_name):
    """Callback to capture mean values at runtime."""
    mean_val = float(mean_tensor.item())
    print(f"[RUNTIME] {layer_name} attention mean: {mean_val:.6f}")
    return mean_tensor  # Return the tensor to keep it in the graph


class AttentionMeanPass(InductorPass):
    """
    A custom inductor pass that calculates the mean of attention outputs.

    This pass finds all unified_attention_with_output operations in the graph
    and injects a mean calculation after each one.
    """

    def __init__(self):
        super().__init__()
        self.attention_op = torch.ops.vllm.unified_attention_with_output.default

    def __call__(self, graph: torch.fx.Graph) -> None:
        """
        Process the graph to inject mean calculations after attention operations.

        Args:
            graph: The FX graph to process
        """
        # Print a summary of the graph
        print(f"\n{'=' * 80}")
        print("Graph Summary")
        print(f"{'=' * 80}")
        print(f"Total nodes: {len(list(graph.nodes))}")

        # Print tabular view (requires: pip install tabulate)
        try:
            graph.print_tabular()
        except ImportError:
            # Fallback to string representation
            print(graph)

        print(f"{'=' * 80}\n")

        # Save the graph to a file
        with open("graph_before.txt", "w") as f:
            f.write(str(graph))

        # Keep track of nodes to avoid modifying graph while iterating
        nodes_to_process = []

        # Find all unified_attention_with_output nodes
        # How model-specific is this?
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.auto_functionalized
            ):
                # Check if it's the attention operation
                if (
                    node.args
                    and str(node.args[0])
                    == "vllm.unified_attention_with_output.default"
                ):
                    nodes_to_process.append(node)

        print(f"Found {len(nodes_to_process)} attention nodes to process")

        # Process each attention node
        for attention_node in nodes_to_process:
            self._inject_mean_calculation(graph, attention_node)

        with open("graph_after.txt", "w") as f:
            f.write(str(graph))

    def _inject_mean_calculation(
        self, graph: torch.fx.Graph, attention_node: fx.Node
    ) -> None:
        """
        Inject mean calculation after an attention node.

        Args:
            graph: The FX graph
            attention_node: The attention node to process
        """
        # The unified_attention_with_output returns a tuple where the second element
        # is the attention output tensor

        layer_name = attention_node.kwargs.get("layer_name", "unknown")

        # Find nodes that use the attention output (second element of the tuple)
        attention_output_users = []

        for user in attention_node.users:
            if (
                user.op == "call_function"
                and user.target == operator.getitem
                and len(user.args) >= 2
                and user.args[1] == 1
            ):  # Getting index 1 (attention output)
                attention_output_users.append(user)

        # For each attention output usage, inject a mean calculation
        for att_output_node in attention_output_users:
            with graph.inserting_after(att_output_node):
                # Calculate mean
                mean_node = graph.call_function(
                    torch.ops.aten.mean.default, args=(att_output_node,), kwargs={}
                )

                capture_node = graph.call_function(
                    capture_mean, args=(mean_node, layer_name), kwargs={}
                )

                # # Create a simple print-like operation to "use" the mean
                # # In practice, you might store this somewhere or use it differently
                # # For demonstration, we'll add a detach operation to ensure it's computed
                # detached_node = graph.call_function(
                #     torch.ops.aten.detach.default, args=(mean_node,), kwargs={}
                # )

                # # Print using torch's print (works in compiled graphs)
                # graph.call_function(
                #     torch.ops.aten._print.default,
                #     args=(f"{layer_name} mean:", detached_node),
                #     kwargs={},
                # )

                print(
                    f"Injected mean calculation after attention output at {att_output_node.name}"
                )


def create_attention_mean_pass():
    return AttentionMeanPass()
