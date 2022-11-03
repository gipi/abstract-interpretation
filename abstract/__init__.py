"""
For now we are using the ghidra_bridge to connect to the
analysis part.
"""
import ghidra_bridge

bridge = ghidra_bridge.GhidraBridge(
    namespace=globals(),
    hook_import=True,  # this allows to import "ghidra"'s packages
)

