# Abstract interpretation library

This library is an attempt to use the formal tools provided
by abstract interpretation (in particular from the book "Principles of
abstract interpretation") while reverse engineering.

This library uses ghidra as a "backend" to extract the control flow
graph information and attempts to rebuild the code structure like
the ghidra's own decompiler (this because ghidra doesn't expose APIs
to extract the structure of the code but only its CFG).

For starter we are going to reconstruct the following "language"

$$
\begin{array} {rlll}
\mathsf{S} & ::= & \mathsf{x = A} \\\\
  & |   & ; \\\\
  & |   & \mathsf{{\bf if}\\, (B)\\, S} \\\\
  & |   & \mathsf{{\bf if}\\, (B)\\, S\\, {\bf else}\, S} \\\\
  & |   & \mathsf{{\bf while}\\, (B)\\, S} \\\\
  & |   & \mathsf{\\{ Sl \\} }    \\\\
\end{array}
$$


## Tests

```
$ python3 -m pytest -v -s --log-cli-level=DEBUG
```
